"""Leakage-conscious walk-forward evaluation built on the F39 backtest engine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite, sqrt
from statistics import mean, median, pstdev
from typing import Generic, TypeVar, overload

from backtest.engine import BacktestConfig, BacktestResult, Bar, run_backtest
from backtest.walk_forward.splits import TemporalSplit, WalkForwardConfig, make_splits

ModelT = TypeVar("ModelT")
FitFunction = Callable[[tuple[Bar, ...]], ModelT]
PredictFunction = Callable[[ModelT, Sequence[Bar]], int]


class _HistoryView(Sequence[Bar]):
    """Read-only prefix view whose public indexing cannot expose future bars."""

    __slots__ = ("_bars", "_stop")

    def __init__(self, bars: tuple[Bar, ...], stop: int) -> None:
        self._bars = bars
        self._stop = stop

    def __len__(self) -> int:
        return self._stop

    @overload
    def __getitem__(self, index: int) -> Bar: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Bar, ...]: ...

    def __getitem__(self, index: int | slice) -> Bar | tuple[Bar, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._stop)
            return tuple(self._bars[position] for position in range(start, stop, step))
        normalized = index + self._stop if index < 0 else index
        if normalized < 0 or normalized >= self._stop:
            raise IndexError("history index out of range")
        return self._bars[normalized]

    def __iter__(self) -> Iterator[Bar]:
        for index in range(self._stop):
            yield self._bars[index]


@dataclass(frozen=True, slots=True)
class RegimeMetrics:
    """Per-observation return diagnostics grouped by externally supplied regime labels."""

    label: str
    observations: int
    mean_period_return: float
    period_volatility: float
    positive_period_fraction: float


@dataclass(frozen=True, slots=True)
class WindowResult:
    """Metrics and complete F39 backtest output for one out-of-sample window."""

    number: int
    split: TemporalSplit
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    backtest: BacktestResult
    annualized_volatility: float
    regime_metrics: tuple[RegimeMetrics, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardSummary:
    """Across-window summaries; returns are not compounded across disjoint test windows."""

    window_count: int
    mean_return: float
    median_return: float
    return_dispersion: float
    mean_annualized_volatility: float
    mean_max_drawdown: float
    mean_sharpe_ratio: float
    mean_sortino_ratio: float
    total_turnover: float
    total_trades: int
    total_commissions: float
    total_slippage_costs: float
    regime_metrics: tuple[RegimeMetrics, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Reproducible result metadata, per-window results, summaries and SHA-256 fingerprint."""

    config: WalkForwardConfig
    backtest_config: BacktestConfig
    strategy_version: str
    windows: tuple[WindowResult, ...]
    summary: WalkForwardSummary
    fingerprint: str


def _period_returns(result: BacktestResult) -> tuple[float, ...]:
    returns: list[float] = []
    previous = result.initial_capital
    for point in result.equity_curve:
        if previous > 0:
            period_return = point.equity / previous - 1.0
            if isfinite(period_return):
                returns.append(period_return)
        previous = point.equity
    return tuple(returns)


def _annualized_volatility(returns: tuple[float, ...], periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * sqrt(periods_per_year)


def _regime_metrics(
    result: BacktestResult,
    labels: Mapping[datetime, str],
    timestamps: tuple[datetime, ...],
) -> tuple[RegimeMetrics, ...]:
    values_by_label: dict[str, list[float]] = defaultdict(list)
    previous = result.initial_capital
    for point, timestamp in zip(result.equity_curve, timestamps, strict=True):
        if previous > 0:
            period_return = point.equity / previous - 1.0
            if isfinite(period_return):
                values_by_label[labels[timestamp]].append(period_return)
        previous = point.equity

    output: list[RegimeMetrics] = []
    for label in sorted(values_by_label):
        values = values_by_label[label]
        output.append(
            RegimeMetrics(
                label=label,
                observations=len(values),
                mean_period_return=mean(values),
                period_volatility=pstdev(values) if len(values) > 1 else 0.0,
                positive_period_fraction=sum(value > 0 for value in values) / len(values),
            )
        )
    return tuple(output)


def _aggregate_regimes(
    groups: Mapping[str, list[float]],
) -> tuple[RegimeMetrics, ...]:
    output: list[RegimeMetrics] = []
    for label in sorted(groups):
        values = groups[label]
        output.append(
            RegimeMetrics(
                label=label,
                observations=len(values),
                mean_period_return=mean(values),
                period_volatility=pstdev(values) if len(values) > 1 else 0.0,
                positive_period_fraction=sum(value > 0 for value in values) / len(values),
            )
        )
    return tuple(output)


def run_walk_forward(
    bars: Sequence[Bar],
    config: WalkForwardConfig,
    *,
    fit: FitFunction[ModelT],
    predict: PredictFunction[ModelT],
    backtest_config: BacktestConfig | None = None,
    regime_by_timestamp: Mapping[datetime, str] | None = None,
    strategy_version: str = "unspecified",
) -> WalkForwardResult:
    """Fit only each training slice and predict each OOS bar from its causal prefix.

    fit receives an immutable tuple containing training rows only. predict receives a
    read-only history view ending at the current bar, including no future bars. Its
    integer target becomes a close-time signal and the F39 engine executes it no earlier
    than the next bar's open. Each OOS window is independently backtested from flat cash;
    any signal on its final bar cannot execute inside that window.
    """
    series = tuple(bars)
    if len(series) < 2:
        raise ValueError("at least two bars are required")
    if not all(isinstance(bar, Bar) for bar in series):
        raise ValueError("bars must contain only Bar instances")
    try:
        ordered = all(
            left.timestamp < right.timestamp
            for left, right in zip(series, series[1:], strict=False)
        )
    except TypeError as exc:
        raise ValueError("bar timestamps must be mutually comparable") from exc
    if not ordered:
        raise ValueError("bar timestamps must be strictly increasing")
    if not isinstance(strategy_version, str) or not strategy_version.strip():
        raise ValueError("strategy_version must be a non-empty string")

    actual_backtest_config = backtest_config or BacktestConfig()
    splits = make_splits(len(series), config)
    timestamps = tuple(bar.timestamp for bar in series)
    labels = dict(regime_by_timestamp or {})
    known_timestamps = set(timestamps)
    if labels.keys() - known_timestamps:
        raise ValueError("regime labels must reference timestamps present in bars")
    if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
        raise ValueError("regime labels must be non-empty strings")

    all_regime_returns: dict[str, list[float]] = defaultdict(list)
    results: list[WindowResult] = []
    window_returns: list[float] = []
    window_volatility: list[float] = []
    window_drawdowns: list[float] = []
    window_sharpe: list[float] = []
    window_sortino: list[float] = []
    total_turnover = 0.0
    total_trades = 0
    total_commissions = 0.0
    total_slippage = 0.0

    for number, split in enumerate(splits, start=1):
        train_bars = series[split.train_start : split.train_stop]
        test_bars = series[split.test_start : split.test_stop]
        if len(train_bars) < config.initial_train_size or len(test_bars) < 2:
            raise AssertionError("internal error: invalid training or test window")
        if labels and any(bar.timestamp not in labels for bar in test_bars):
            raise ValueError("every out-of-sample timestamp requires a regime label")

        model = fit(train_bars)
        targets: dict[datetime, int] = {}
        for index in range(split.test_start, split.test_stop):
            target = predict(model, _HistoryView(series, index + 1))
            if not isinstance(target, int) or isinstance(target, bool):
                raise ValueError("predict must return an integer signed-unit target")
            targets[series[index].timestamp] = target

        backtest = run_backtest(test_bars, targets, actual_backtest_config)
        returns = _period_returns(backtest)
        annualized_volatility = _annualized_volatility(
            returns, actual_backtest_config.periods_per_year
        )
        test_timestamps = tuple(bar.timestamp for bar in test_bars)
        current_regime_metrics = (
            _regime_metrics(backtest, labels, test_timestamps) if labels else ()
        )
        for item in current_regime_metrics:
            values: list[float] = []
            previous = backtest.initial_capital
            for point, timestamp in zip(backtest.equity_curve, test_timestamps, strict=True):
                if timestamp == timestamps[split.test_start]:
                    values.clear()
                if previous > 0 and labels[timestamp] == item.label:
                    period_return = point.equity / previous - 1.0
                    if isfinite(period_return):
                        values.append(period_return)
                previous = point.equity
            all_regime_returns[item.label].extend(values)

        result = WindowResult(
            number=number,
            split=split,
            train_start=train_bars[0].timestamp,
            train_end=train_bars[-1].timestamp,
            test_start=test_bars[0].timestamp,
            test_end=test_bars[-1].timestamp,
            backtest=backtest,
            annualized_volatility=annualized_volatility,
            regime_metrics=current_regime_metrics,
        )
        results.append(result)
        window_returns.append(backtest.total_return)
        window_volatility.append(annualized_volatility)
        window_drawdowns.append(backtest.max_drawdown)
        window_sharpe.append(backtest.sharpe_ratio)
        window_sortino.append(backtest.sortino_ratio)
        total_turnover += backtest.turnover
        total_trades += backtest.trades
        total_commissions += backtest.commissions
        total_slippage += backtest.slippage_costs

    summary = WalkForwardSummary(
        window_count=len(results),
        mean_return=mean(window_returns),
        median_return=median(window_returns),
        return_dispersion=pstdev(window_returns) if len(window_returns) > 1 else 0.0,
        mean_annualized_volatility=mean(window_volatility),
        mean_max_drawdown=mean(window_drawdowns),
        mean_sharpe_ratio=mean(window_sharpe),
        mean_sortino_ratio=mean(window_sortino),
        total_turnover=total_turnover,
        total_trades=total_trades,
        total_commissions=total_commissions,
        total_slippage_costs=total_slippage,
        regime_metrics=_aggregate_regimes(all_regime_returns),
    )

    effective_config = {
        "initial_train_size": config.initial_train_size,
        "test_size": config.test_size,
        "step_size": config.effective_step_size,
        "mode": config.mode,
        "gap": config.gap,
        "anchor_index": config.anchor_index,
        "allow_partial_test": config.allow_partial_test,
    }
    window_payload = [
        {
            "number": item.number,
            "train_start": item.train_start.isoformat(),
            "train_end": item.train_end.isoformat(),
            "test_start": item.test_start.isoformat(),
            "test_end": item.test_end.isoformat(),
            "return": item.backtest.total_return,
            "volatility": item.annualized_volatility,
            "drawdown": item.backtest.max_drawdown,
            "sharpe": item.backtest.sharpe_ratio,
            "sortino": item.backtest.sortino_ratio,
            "turnover": item.backtest.turnover,
            "trades": item.backtest.trades,
            "commissions": item.backtest.commissions,
            "slippage": item.backtest.slippage_costs,
            "regimes": [asdict(item) for item in item.regime_metrics],
        }
        for item in results
    ]
    payload = {
        "strategy_version": strategy_version,
        "walk_forward_config": effective_config,
        "backtest_config": asdict(actual_backtest_config),
        "bars": [
            [
                bar.timestamp.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ]
            for bar in series
        ],
        "regime_labels": sorted((timestamp.isoformat(), label) for timestamp, label in labels.items()),
        "windows": window_payload,
        "summary": asdict(summary),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = sha256(serialized.encode("utf-8")).hexdigest()

    return WalkForwardResult(
        config=config,
        backtest_config=actual_backtest_config,
        strategy_version=strategy_version,
        windows=tuple(results),
        summary=summary,
        fingerprint=fingerprint,
    )
