"""Deterministic, long/short-unit backtesting with next-bar-open execution.

Signals are observed at a bar close and become eligible for execution at the
next bar's open. This module is a research simulator only; it never sends orders.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from statistics import mean, pstdev


@dataclass(frozen=True, slots=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        prices = (self.open, self.high, self.low, self.close)
        if not all(isfinite(value) and value > 0 for value in prices):
            raise ValueError("OHLC prices must be finite and positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be at least open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be at most open, close and high")
        if not isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if not isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        if not isfinite(self.commission_bps) or self.commission_bps < 0:
            raise ValueError("commission_bps must be finite and non-negative")
        if not isfinite(self.slippage_bps) or self.slippage_bps < 0:
            raise ValueError("slippage_bps must be finite and non-negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


@dataclass(frozen=True, slots=True)
class Fill:
    timestamp: datetime
    units_delta: int
    units_after: int
    reference_price: float
    fill_price: float
    commission: float
    slippage_cost: float


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    cash: float
    units: int
    close: float
    equity: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    turnover: float
    trades: int
    commissions: float
    slippage_costs: float
    equity_curve: tuple[EquityPoint, ...]
    fills: tuple[Fill, ...]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _max_drawdown(equity: Iterable[float]) -> float:
    peak = 0.0
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            drawdown = max(drawdown, (peak - value) / peak)
    return drawdown


def run_backtest(
    bars: Iterable[Bar],
    target_units_by_timestamp: Mapping[datetime, int],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run signals emitted at close and execute them at the following bar open.

    The mapping assigns a desired signed position to a bar timestamp. That target
    becomes actionable only after the bar closes. The final bar's signal is not
    executed because no subsequent open is available.
    """
    settings = config or BacktestConfig()
    series = tuple(bars)
    if len(series) < 2:
        raise ValueError("at least two bars are required")
    timestamps = [bar.timestamp for bar in series]
    if any(
        left >= right
        for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("bar timestamps must be strictly increasing")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in target_units_by_timestamp.values()
    ):
        raise ValueError("target positions must be integer units")
    known = set(timestamps)
    if any(timestamp not in known for timestamp in target_units_by_timestamp):
        raise ValueError("signals must reference an existing bar timestamp")

    cash = settings.initial_capital
    units = 0
    pending_target: int | None = None
    fills: list[Fill] = []
    curve: list[EquityPoint] = []
    commissions = 0.0
    slippage_costs = 0.0
    traded_notional = 0.0

    for bar in series:
        if pending_target is not None and pending_target != units:
            delta = pending_target - units
            reference = bar.open
            direction = 1.0 if delta > 0 else -1.0
            fill_price = reference * (
                1.0 + direction * settings.slippage_bps / 10_000
            )
            notional = abs(delta * fill_price)
            commission = notional * settings.commission_bps / 10_000
            slip_cost = abs(delta) * abs(fill_price - reference)
            cash -= delta * fill_price + commission
            units = pending_target
            commissions += commission
            slippage_costs += slip_cost
            traded_notional += notional
            fills.append(
                Fill(
                    bar.timestamp,
                    delta,
                    units,
                    reference,
                    fill_price,
                    commission,
                    slip_cost,
                )
            )
        equity = cash + units * bar.close
        if not isfinite(equity):
            raise ArithmeticError("non-finite equity encountered")
        curve.append(EquityPoint(bar.timestamp, cash, units, bar.close, equity))
        pending_target = target_units_by_timestamp.get(bar.timestamp)

    values = [point.equity for point in curve]
    returns = [
        current / previous - 1.0
        for previous, current in zip(values, values[1:], strict=False)
        if previous > 0
    ]
    avg_return = mean(returns) if returns else 0.0
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = (
        sqrt(mean(value * value for value in downside)) if downside else 0.0
    )
    sharpe = _ratio(avg_return, volatility) * sqrt(settings.periods_per_year)
    sortino = _ratio(avg_return, downside_deviation) * sqrt(settings.periods_per_year)
    final_equity = values[-1]
    return BacktestResult(
        initial_capital=settings.initial_capital,
        final_equity=final_equity,
        total_return=final_equity / settings.initial_capital - 1.0,
        max_drawdown=_max_drawdown(values),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        turnover=traded_notional / settings.initial_capital,
        trades=len(fills),
        commissions=commissions,
        slippage_costs=slippage_costs,
        equity_curve=tuple(curve),
        fills=tuple(fills),
    )
