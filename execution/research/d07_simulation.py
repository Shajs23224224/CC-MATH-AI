"""D07.2: cost-sensitivity simulation of integrity-audited one-session decisions."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execution.research.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    manifest_output_entries,
    read_json_object,
    sha256_file,
    verify_manifest_outputs,
)
from scripts.run_execution_decision_policy import apply_neutral_threshold
from scripts.run_execution_ml_walk_forward import EXPECTED_CLASSES

_AUDIT_OUTPUTS = (
    "audited_predictions.csv",
    "fold_audit.csv",
    "audit_summary.json",
)
_REQUIRED_PREDICTION_COLUMNS = {
    "algorithm",
    "fold",
    "row_index",
    "feature_date",
    "entry_date",
    "actual",
    "argmax_prediction",
    "policy_prediction",
    "selected_neutral_threshold",
    "probability_down",
    "probability_neutral",
    "probability_up",
    "entry_open",
    "exit_close",
    "actual_execution_return",
}
_STRATEGIES = {
    "argmax": "argmax_prediction",
    "neutral_policy": "policy_prediction",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    _require(np.isfinite(result), f"{field} must be finite")
    return result


def _validate_configuration(
    initial_capital: float,
    notional_per_trade: float,
    cost_scenarios_bps_per_side: tuple[float, ...],
    trading_days_per_year: int,
) -> tuple[float, ...]:
    capital = _finite_number(initial_capital, "initial_capital")
    notional = _finite_number(notional_per_trade, "notional_per_trade")
    _require(capital > 0.0, "initial_capital must be positive")
    _require(
        0.0 < notional <= capital,
        "notional_per_trade must be positive and <= initial_capital",
    )
    _require(
        isinstance(trading_days_per_year, int)
        and not isinstance(trading_days_per_year, bool)
        and 1 <= trading_days_per_year <= 366,
        "trading_days_per_year must be an integer between 1 and 366",
    )
    _require(
        bool(cost_scenarios_bps_per_side),
        "at least one transaction-cost scenario is required",
    )
    costs = tuple(
        _finite_number(value, "cost_bps_per_side") for value in cost_scenarios_bps_per_side
    )
    _require(
        len(set(costs)) == len(costs),
        "cost scenarios must not contain duplicates",
    )
    _require(
        all(0.0 <= value <= 1000.0 for value in costs),
        "each cost scenario must be between 0 and 1000 bps per side",
    )
    return tuple(sorted(costs))


def _load_verified_audit(
    audit_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    audit_dir = audit_dir.expanduser().resolve()
    _require(audit_dir.is_dir(), f"D07.1 audit directory not found: {audit_dir}")
    manifest, _ = verify_manifest_outputs(
        audit_dir,
        "audit_manifest.json",
        _AUDIT_OUTPUTS,
    )
    summary = read_json_object(audit_dir / "audit_summary.json")
    _require(
        summary.get("status") == "COMPLETED"
        and summary.get("audit_type") == "D07.1_D06_prediction_integrity",
        "D07.1 audit is not completed or has an unknown schema",
    )
    checks = summary.get("checks")
    checks_total = summary.get("checks_total")
    checks_passed = summary.get("checks_passed")
    _require(
        isinstance(checks, list)
        and isinstance(checks_total, int)
        and isinstance(checks_passed, int)
        and len(checks) == checks_total
        and checks_total > 0
        and checks_passed == checks_total
        and all(isinstance(item, dict) and item.get("passed") is True for item in checks),
        "D07.1 contains failed or incomplete integrity checks",
    )
    _require(
        summary.get("algorithm_count") == len(summary.get("algorithms", [])),
        "D07.1 algorithm count does not match its algorithm list",
    )
    input_manifest = manifest.get("input")
    _require(isinstance(input_manifest, dict), "D07.1 manifest input is missing")
    for key in (
        "decision_policy_run_id",
        "execution_dataset_sha256",
        "source_manifest_sha256",
        "d06_manifest_sha256",
    ):
        _require(
            input_manifest.get(key) == summary.get(key),
            f"D07.1 manifest and summary disagree on {key}",
        )

    predictions = pd.read_csv(
        audit_dir / "audited_predictions.csv",
        float_precision="round_trip",
    )
    fold_audit = pd.read_csv(audit_dir / "fold_audit.csv")
    _require(
        len(predictions) == summary.get("rows_audited"),
        "D07.1 prediction row count differs from summary",
    )
    return predictions, fold_audit, summary, manifest


def _validate_audited_rows(
    predictions: pd.DataFrame,
    fold_audit: pd.DataFrame,
    summary: dict[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...], int]:
    missing = _REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    _require(not missing, f"D07.1 predictions missing columns: {sorted(missing)}")
    required_fold = {
        "algorithm",
        "fold",
        "train_start_index",
        "train_end_index",
        "test_start_index",
        "test_end_index",
        "test_samples",
        "selected_neutral_threshold",
    }
    _require(
        required_fold.issubset(fold_audit.columns),
        "D07.1 fold audit schema is incomplete",
    )
    _require(not predictions.empty, "D07.1 predictions are empty")
    _require(not fold_audit.empty, "D07.1 fold audit is empty")

    raw_algorithms = summary.get("algorithms")
    _require(
        isinstance(raw_algorithms, list)
        and bool(raw_algorithms)
        and all(isinstance(value, str) and value for value in raw_algorithms)
        and len(set(raw_algorithms)) == len(raw_algorithms),
        "D07.1 algorithm list is invalid",
    )
    algorithms = tuple(raw_algorithms)
    fold_count = summary.get("fold_count")
    _require(
        isinstance(fold_count, int)
        and not isinstance(fold_count, bool)
        and fold_count > 0,
        "D07.1 fold count is invalid",
    )
    _require(
        set(predictions["algorithm"].astype(str)) == set(algorithms),
        "D07.1 algorithm names differ from summary",
    )
    _require(
        set(fold_audit["algorithm"].astype(str)) == set(algorithms),
        "D07.1 fold-audit algorithms differ from summary",
    )
    _require(
        len(fold_audit) == len(algorithms) * fold_count,
        "D07.1 fold-audit row count differs from summary",
    )
    _require(
        not predictions.duplicated(["algorithm", "row_index"]).any(),
        "D07.1 predictions contain duplicate algorithm/row_index pairs",
    )
    _require(
        not fold_audit.duplicated(["algorithm", "fold"]).any(),
        "D07.1 fold audit contains duplicate algorithm/fold pairs",
    )

    for column in ("fold", "row_index"):
        values = pd.to_numeric(predictions[column], errors="raise").to_numpy(dtype=float)
        _require(np.all(np.isfinite(values)), f"D07.1 {column} contains non-finite values")
        _require(np.all(values == np.floor(values)), f"D07.1 {column} must be integer")
        if column == "row_index":
            _require(np.all(values >= 0.0), "D07.1 row_index must be non-negative")
        predictions[column] = values.astype(np.int64)

    fold_audit = fold_audit.copy()
    for column in (
        "fold",
        "train_start_index",
        "train_end_index",
        "test_start_index",
        "test_end_index",
        "test_samples",
    ):
        values = pd.to_numeric(fold_audit[column], errors="raise").to_numpy(dtype=float)
        _require(np.all(np.isfinite(values)), f"fold audit {column} is non-finite")
        _require(np.all(values == np.floor(values)), f"fold audit {column} must be integer")
        fold_audit[column] = values.astype(np.int64)

    numeric_columns = (
        "entry_open",
        "exit_close",
        "actual_execution_return",
        "selected_neutral_threshold",
        "probability_down",
        "probability_neutral",
        "probability_up",
    )
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    numeric = predictions[list(numeric_columns)].to_numpy(dtype=np.float64)
    _require(np.all(np.isfinite(numeric)), "D07.1 predictions contain non-finite numeric values")
    _require(np.all(numeric[:, :2] > 0.0), "D07.1 execution prices must be positive")
    calculated_returns = (
        predictions["exit_close"].to_numpy(dtype=float)
        / predictions["entry_open"].to_numpy(dtype=float)
        - 1.0
    )
    _require(
        bool(np.allclose(
            predictions["actual_execution_return"].to_numpy(dtype=float),
            calculated_returns,
            rtol=1e-10,
            atol=1e-12,
        )),
        "D07.1 realized returns do not match execution prices",
    )

    probability_columns = ["probability_down", "probability_neutral", "probability_up"]
    probabilities = predictions[probability_columns].to_numpy(dtype=np.float64)
    _require(
        np.all(probabilities >= 0.0)
        and np.all(probabilities <= 1.0)
        and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
        "D07.1 contains invalid probability vectors",
    )
    thresholds = predictions["selected_neutral_threshold"].to_numpy(dtype=float)
    _require(
        np.all((thresholds >= 0.0) & (thresholds <= 1.0)),
        "D07.1 selected thresholds must be in [0, 1]",
    )

    for column in ("actual", "argmax_prediction", "policy_prediction"):
        predictions[column] = predictions[column].astype(str)
        _require(
            set(predictions[column]).issubset(EXPECTED_CLASSES),
            f"D07.1 {column} contains an unknown class",
        )
    for column in ("feature_date", "entry_date"):
        parsed = pd.to_datetime(predictions[column], errors="coerce", utc=True)
        _require(not parsed.isna().any(), f"D07.1 {column} contains invalid dates")
        predictions[f"_{column}_utc"] = parsed
    _require(
        bool((predictions["_entry_date_utc"] > predictions["_feature_date_utc"]).all()),
        "D07.1 entry dates must follow feature dates",
    )
    _require(
        predictions["_feature_date_utc"].is_monotonic_increasing
        and predictions["_entry_date_utc"].is_monotonic_increasing,
        "D07.1 feature/entry dates must be chronological",
    )
    _require(
        not predictions.duplicated(["algorithm", "_feature_date_utc"]).any(),
        "D07.1 feature dates contain duplicates",
    )
    _require(
        not predictions.duplicated(["algorithm", "_entry_date_utc"]).any(),
        "D07.1 entry dates contain duplicates",
    )

    canonical_indices: np.ndarray | None = None
    canonical_folds: np.ndarray | None = None
    canonical_dates: list[tuple[int, int]] | None = None
    canonical_prices: np.ndarray | None = None

    for algorithm in algorithms:
        selected = predictions.loc[
            predictions["algorithm"].astype(str) == algorithm
        ].sort_values(["fold", "row_index"])
        _require(not selected.empty, f"D07.1 has no rows for {algorithm}")
        _require(
            selected["_feature_date_utc"].is_monotonic_increasing
            and selected["_entry_date_utc"].is_monotonic_increasing,
            f"{algorithm} feature/entry dates are not chronological",
        )
        fold_ids = selected["fold"].to_numpy(dtype=np.int64)
        _require(
            set(fold_ids.tolist()) == set(range(1, fold_count + 1)),
            f"D07.1 fold identifiers are incomplete for {algorithm}",
        )
        indices = selected["row_index"].to_numpy(dtype=np.int64)
        dates = list(
            zip(
                selected["_feature_date_utc"].astype("int64").tolist(),
                selected["_entry_date_utc"].astype("int64").tolist(),
                strict=True,
            )
        )
        prices = selected[
            ["entry_open", "exit_close", "actual_execution_return"]
        ].to_numpy(dtype=np.float64)
        if canonical_indices is None:
            canonical_indices = indices
            canonical_folds = fold_ids
            canonical_dates = dates
            canonical_prices = prices
        else:
            _require(
                np.array_equal(indices, canonical_indices),
                "algorithms do not share identical OOS row indexes",
            )
            _require(
                np.array_equal(fold_ids, canonical_folds),
                "algorithms do not share identical fold assignments",
            )
            _require(dates == canonical_dates, "algorithms do not share identical OOS dates")
            _require(
                bool(np.allclose(prices, canonical_prices, rtol=1e-10, atol=1e-12)),
                "algorithms do not share identical realized execution outcomes",
            )

        algorithm_fold_audit = fold_audit.loc[
            fold_audit["algorithm"].astype(str) == algorithm
        ].sort_values("fold")
        _require(
            algorithm_fold_audit["fold"].tolist() == list(range(1, fold_count + 1)),
            f"fold-audit identifiers are incomplete for {algorithm}",
        )
        previous_end: int | None = None
        for audit_row in algorithm_fold_audit.itertuples(index=False):
            start = int(audit_row.test_start_index)
            end = int(audit_row.test_end_index)
            _require(
                start >= 0
                and end > start
                and (previous_end is None or start == previous_end),
                f"{algorithm} fold audit has a gap or overlapping OOS windows",
            )
            _require(
                int(audit_row.train_start_index) == 0
                and int(audit_row.train_end_index) == start,
                f"{algorithm} fold audit has invalid training boundaries",
            )
            _require(
                int(audit_row.test_samples) == end - start,
                f"{algorithm} fold-audit sample count mismatch",
            )
            selected_fold = selected.loc[selected["fold"] == int(audit_row.fold)]
            expected_indices = np.arange(start, end, dtype=np.int64)
            _require(
                np.array_equal(
                    selected_fold["row_index"].to_numpy(dtype=np.int64),
                    expected_indices,
                ),
                f"{algorithm} fold {audit_row.fold} row coverage mismatch",
            )
            _require(
                selected_fold["selected_neutral_threshold"].nunique(dropna=False) == 1
                and np.isclose(
                    float(selected_fold["selected_neutral_threshold"].iloc[0]),
                    float(audit_row.selected_neutral_threshold),
                    rtol=0.0,
                    atol=1e-12,
                ),
                f"{algorithm} fold {audit_row.fold} threshold mismatch",
            )
            previous_end = end

        expected_argmax = np.asarray(EXPECTED_CLASSES, dtype=str)[
            np.argmax(
                selected[probability_columns].to_numpy(dtype=np.float64),
                axis=1,
            )
        ]
        _require(
            np.array_equal(selected["argmax_prediction"].astype(str).to_numpy(), expected_argmax),
            f"{algorithm} argmax predictions cannot be reproduced",
        )
        # Recompute the threshold policy independently within each calibrated fold.
        for _, fold_rows in selected.groupby("fold", sort=True):
            threshold = float(fold_rows["selected_neutral_threshold"].iloc[0])
            expected = np.asarray(
                apply_neutral_threshold(
                    fold_rows[probability_columns].to_numpy(dtype=np.float64),
                    threshold,
                ),
                dtype=str,
            )
            actual_policy = fold_rows["policy_prediction"].astype(str).to_numpy()
            _require(
                np.array_equal(actual_policy, expected),
                f"{algorithm} policy predictions cannot be reproduced",
            )

    _require(
        len(predictions) == int(summary.get("rows_audited", -1)),
        "D07.1 summary prediction count mismatch",
    )
    predictions = predictions.sort_values(
        ["algorithm", "fold", "row_index"]
    ).reset_index(drop=True)
    return predictions, algorithms, fold_count


def _simulate_account(
    frame: pd.DataFrame,
    *,
    algorithm: str,
    strategy: str,
    prediction_column: str,
    initial_capital: float,
    notional_per_trade: float,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    ordered = frame.sort_values(["fold", "row_index"]).reset_index(drop=True)
    equity = float(initial_capital)
    peak_equity = float(initial_capital)
    bankrupt = False
    round_trip_cost_rate = 2.0 * cost_bps_per_side / 10_000.0
    records: list[dict[str, Any]] = []

    for row in ordered.itertuples(index=False):
        action = str(getattr(row, prediction_column))
        requested_direction = 1 if action == "UP" else -1 if action == "DOWN" else 0
        equity_before = equity
        active_notional = 0.0
        gross_pnl = 0.0
        costs = 0.0
        net_pnl = 0.0
        executed_direction = 0

        if bankrupt:
            status = "SKIPPED_BANKRUPTCY"
        elif requested_direction == 0:
            status = "NO_TRADE"
        else:
            executed_direction = requested_direction
            active_notional = min(notional_per_trade, equity_before)
            gross_pnl = (
                executed_direction
                * float(row.actual_execution_return)
                * active_notional
            )
            costs = active_notional * round_trip_cost_rate
            net_pnl = gross_pnl - costs
            equity += net_pnl
            status = "EXECUTED"
            if equity <= 0.0:
                bankrupt = True
                status = "PORTFOLIO_BROKE"

        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0
        records.append(
            {
                "algorithm": algorithm,
                "strategy": strategy,
                "cost_bps_per_side": float(cost_bps_per_side),
                "round_trip_cost_bps": float(2.0 * cost_bps_per_side),
                "fold": int(row.fold),
                "row_index": int(row.row_index),
                "feature_date": str(row.feature_date),
                "entry_date": str(row.entry_date),
                "signal": action,
                "requested_direction": requested_direction,
                "executed_direction": executed_direction,
                "execution_status": status,
                "actual_execution_return": float(row.actual_execution_return),
                "entry_open": float(row.entry_open),
                "exit_close": float(row.exit_close),
                "configured_notional": float(notional_per_trade),
                "active_notional": float(active_notional),
                "gross_pnl": float(gross_pnl),
                "transaction_costs": float(costs),
                "net_pnl": float(net_pnl),
                "net_return_on_active_notional": (
                    float(net_pnl / active_notional) if active_notional > 0.0 else 0.0
                ),
                "portfolio_equity_before": float(equity_before),
                "portfolio_equity_after": float(equity),
                "drawdown": float(drawdown),
                "bankrupt_after_session": bool(bankrupt),
            }
        )
    return pd.DataFrame.from_records(records)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def _account_metrics(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
    trading_days_per_year: int,
) -> dict[str, Any]:
    pnl = frame["net_pnl"].to_numpy(dtype=np.float64)
    gross_pnl = frame["gross_pnl"].to_numpy(dtype=np.float64)
    costs = frame["transaction_costs"].to_numpy(dtype=np.float64)
    directions = frame["executed_direction"].to_numpy(dtype=np.int64)
    traded = directions != 0
    trade_pnl = pnl[traded]
    daily_returns = pnl / initial_capital
    equity = frame["portfolio_equity_after"].to_numpy(dtype=np.float64)
    running_peak = np.maximum.accumulate(
        np.concatenate(([initial_capital], equity))
    )[1:]
    drawdowns = equity / running_peak - 1.0
    winners = trade_pnl[trade_pnl > 0.0]
    losers = trade_pnl[trade_pnl < 0.0]
    total_notional = float(frame["active_notional"].sum())
    sessions = len(frame)
    trade_count = int(traded.sum())
    return_std = float(np.std(daily_returns, ddof=1)) if sessions > 1 else 0.0
    downside_deviation = float(np.sqrt(np.mean(np.minimum(daily_returns, 0.0) ** 2)))

    return {
        "sessions": sessions,
        "trades": trade_count,
        "long_trades": int(np.sum(directions == 1)),
        "short_trades": int(np.sum(directions == -1)),
        "neutral_signals": int((frame["signal"] == "NEUTRAL").sum()),
        "skipped_bankruptcy_sessions": int(
            (frame["execution_status"] == "SKIPPED_BANKRUPTCY").sum()
        ),
        "exposure_ratio": float(trade_count / sessions) if sessions else 0.0,
        "initial_capital": float(initial_capital),
        "final_equity": float(equity[-1]),
        "total_gross_pnl": float(gross_pnl.sum()),
        "total_transaction_costs": float(costs.sum()),
        "total_net_pnl": float(pnl.sum()),
        "simple_total_return": float(pnl.sum() / initial_capital),
        "simple_annualized_return": float(
            pnl.sum() / initial_capital * trading_days_per_year / sessions
        ),
        "annualized_volatility": float(return_std * math.sqrt(trading_days_per_year)),
        "sharpe_ratio": _ratio(
            float(np.mean(daily_returns)) * math.sqrt(trading_days_per_year),
            return_std,
        ),
        "sortino_ratio": _ratio(
            float(np.mean(daily_returns)) * math.sqrt(trading_days_per_year),
            downside_deviation,
        ),
        "maximum_drawdown": float(np.min(drawdowns)),
        "winning_trades": int(len(winners)),
        "losing_trades": int(len(losers)),
        "breakeven_trades": int(trade_count - len(winners) - len(losers)),
        "win_rate": _ratio(float(len(winners)), float(trade_count)),
        "average_trade_net_pnl": float(np.mean(trade_pnl)) if trade_count else None,
        "profit_factor": _ratio(float(winners.sum()), float(abs(losers.sum()))),
        "one_way_traded_notional": total_notional,
        "round_trip_turnover_notional": 2.0 * total_notional,
        "break_even_cost_bps_per_side": _ratio(
            float(gross_pnl.sum()) * 10_000.0,
            2.0 * total_notional,
        ),
        "bankrupt": bool(frame["bankrupt_after_session"].any()),
        "bankruptcy_date": (
            str(frame.loc[frame["execution_status"] == "PORTFOLIO_BROKE", "entry_date"].iloc[0])
            if (frame["execution_status"] == "PORTFOLIO_BROKE").any()
            else None
        ),
        "return_basis": "simple daily PnL / initial capital; no compounding",
    }


def _fold_metrics(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, group in frame.groupby("fold", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        pnl = group["net_pnl"].to_numpy(dtype=np.float64)
        traded = group["executed_direction"].to_numpy(dtype=np.int64) != 0
        starting_equity = float(first["portfolio_equity_before"])
        ending_equity = float(last["portfolio_equity_after"])
        rows.append(
            {
                "algorithm": str(first["algorithm"]),
                "strategy": str(first["strategy"]),
                "cost_bps_per_side": float(first["cost_bps_per_side"]),
                "fold": int(fold),
                "start_row_index": int(group["row_index"].min()),
                "end_row_index_exclusive": int(group["row_index"].max()) + 1,
                "sessions": int(len(group)),
                "trades": int(traded.sum()),
                "neutral_signals": int((group["signal"] == "NEUTRAL").sum()),
                "skipped_bankruptcy_sessions": int(
                    (group["execution_status"] == "SKIPPED_BANKRUPTCY").sum()
                ),
                "gross_pnl": float(group["gross_pnl"].sum()),
                "transaction_costs": float(group["transaction_costs"].sum()),
                "net_pnl": float(pnl.sum()),
                "equity_at_fold_start": starting_equity,
                "equity_at_fold_end": ending_equity,
                "return_on_initial_capital": float(pnl.sum() / initial_capital),
                "return_on_fold_start_equity": _ratio(
                    float(pnl.sum()),
                    starting_equity,
                ),
                "minimum_drawdown": float(group["drawdown"].min()),
                "bankrupt_at_fold_end": bool(group["bankrupt_after_session"].any()),
            }
        )
    return rows


def simulate_execution_decisions(
    audit_dir: Path,
    *,
    output_root: Path = Path("artifacts/execution_simulations"),
    initial_capital: float = 100_000.0,
    notional_per_trade: float = 10_000.0,
    cost_scenarios_bps_per_side: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    """Run independent fixed-notional accounts across cost scenarios.

    Every prediction represents one next-session open-to-close position.
    UP is long, DOWN is short and NEUTRAL is flat. The per-side cost is charged
    on entry and exit. No broker orders, leverage or compounding are used.
    """
    costs = _validate_configuration(
        initial_capital,
        notional_per_trade,
        cost_scenarios_bps_per_side,
        trading_days_per_year,
    )
    audit_dir = audit_dir.expanduser().resolve()
    predictions, fold_audit, audit_summary, _ = _load_verified_audit(audit_dir)
    predictions, algorithms, fold_count = _validate_audited_rows(
        predictions,
        fold_audit,
        audit_summary,
    )

    ledgers: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        algorithm_rows = predictions.loc[
            predictions["algorithm"].astype(str) == algorithm
        ]
        for strategy, prediction_column in _STRATEGIES.items():
            for cost in costs:
                ledger = _simulate_account(
                    algorithm_rows,
                    algorithm=algorithm,
                    strategy=strategy,
                    prediction_column=prediction_column,
                    initial_capital=initial_capital,
                    notional_per_trade=notional_per_trade,
                    cost_bps_per_side=cost,
                )
                ledgers.append(ledger)
                metric: dict[str, Any] = {
                    "algorithm": algorithm,
                    "strategy": strategy,
                    "cost_bps_per_side": cost,
                    "round_trip_cost_bps": 2.0 * cost,
                    "configured_notional_per_trade": float(notional_per_trade),
                    "trading_days_per_year": trading_days_per_year,
                }
                metric.update(
                    _account_metrics(
                        ledger,
                        initial_capital=initial_capital,
                        trading_days_per_year=trading_days_per_year,
                    )
                )
                metric_rows.append(metric)
                fold_rows.extend(
                    _fold_metrics(ledger, initial_capital=initial_capital)
                )

    trade_ledger = pd.concat(ledgers, ignore_index=True).sort_values(
        ["algorithm", "strategy", "cost_bps_per_side", "fold", "row_index"]
    ).reset_index(drop=True)
    metrics = pd.DataFrame.from_records(metric_rows)
    folds = pd.DataFrame.from_records(fold_rows)
    sensitivity_columns = [
        "algorithm",
        "strategy",
        "cost_bps_per_side",
        "round_trip_cost_bps",
        "trades",
        "total_gross_pnl",
        "total_transaction_costs",
        "total_net_pnl",
        "simple_total_return",
        "maximum_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "break_even_cost_bps_per_side",
    ]
    sensitivity = metrics[sensitivity_columns].copy()

    ticker = str(audit_summary.get("ticker", "UNKNOWN"))
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]+", "_", ticker).lower()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    simulation_id = f"{safe_ticker}_d07_2_{timestamp}"
    simulation_dir = output_root / simulation_id
    simulation_dir.mkdir(parents=False, exist_ok=False)

    ledger_path = simulation_dir / "trade_ledger.csv"
    metrics_path = simulation_dir / "strategy_metrics.csv"
    folds_path = simulation_dir / "fold_metrics.csv"
    sensitivity_path = simulation_dir / "cost_sensitivity.csv"
    summary_path = simulation_dir / "simulation_summary.json"
    atomic_write_csv(ledger_path, trade_ledger)
    atomic_write_csv(metrics_path, metrics)
    atomic_write_csv(folds_path, folds)
    atomic_write_csv(sensitivity_path, sensitivity)

    summary = {
        "schema_version": "2.0",
        "status": "COMPLETED",
        "simulation_type": "D07.2_cost_sensitivity_one_session",
        "created_at": datetime.now(UTC).isoformat(),
        "simulation_id": simulation_id,
        "ticker": ticker,
        "audit_directory": audit_dir.name,
        "audit_manifest_sha256": sha256_file(audit_dir / "audit_manifest.json"),
        "audited_predictions_sha256": sha256_file(audit_dir / "audited_predictions.csv"),
        "decision_policy_run_id": audit_summary.get("decision_policy_run_id"),
        "execution_dataset_sha256": audit_summary.get("execution_dataset_sha256"),
        "algorithm_count": len(algorithms),
        "algorithms": list(algorithms),
        "fold_count": fold_count,
        "strategies": list(_STRATEGIES),
        "cost_scenarios_bps_per_side": list(costs),
        "sessions_per_algorithm": int(
            predictions.groupby("algorithm")["row_index"].nunique().iloc[0]
        ),
        "accounts_simulated": len(metric_rows),
        "capital_model": {
            "initial_capital": float(initial_capital),
            "notional_per_trade": float(notional_per_trade),
            "notional_cap": "min(configured notional, current equity before trade)",
            "account_scope": "independent for each algorithm, strategy and cost scenario",
            "pnl_accounting": "fixed-notional additive PnL; profits are not reinvested",
            "post_bankruptcy": "no additional positions are opened in that account",
        },
        "execution_model": {
            "signal_time": "after feature session close",
            "entry": "recorded next-session open",
            "exit": "recorded same-session close",
            "holding_period": "one session; consecutive positions do not overlap",
            "UP": "long",
            "DOWN": "short",
            "NEUTRAL": "flat; no transaction cost",
            "orders_routed": False,
        },
        "transaction_cost_model": {
            "cost_bps_per_side": list(costs),
            "round_trip_cost_bps": [2.0 * value for value in costs],
            "cost_basis": "all-in per-leg estimate applied on entry and exit",
            "included_in_estimate": "commissions, spread and slippage",
            "not_independently_modeled": [
                "market impact",
                "borrow fees and short availability",
                "funding and financing",
                "taxes",
                "dividends and corporate actions",
                "partial fills and latency",
                "broker margin and liquidation rules",
            ],
        },
        "metric_interpretation": {
            "simple_annualized_return": "linear annualization, not CAGR",
            "sharpe_sortino": "calculated from daily net PnL / initial capital",
            "maximum_drawdown": "marked from initial capital and subsequent equity highs",
            "break_even_cost_bps_per_side": (
                "gross PnL / round-trip notional, converted to bps per side"
            ),
        },
        "limitations": [
            "Historical simulation is not live trading; recorded prices are theoretical fills.",
            (
                "A short can lose more than its notional; negative equity is "
                "recorded and later trades are stopped."
            ),
            "The fixed-notional model does not reinvest profits or compound returns.",
            (
                "Borrow fees, availability, dividends, financing, taxes, market "
                "impact and broker margin rules are not modeled."
            ),
            (
                "Historical OOS folds were previously inspected and are not an "
                "untouched prospective test."
            ),
            (
                "Cost scenarios are assumptions; results do not establish future "
                "profitability or suitability for capital deployment."
            ),
        ],
        "outputs": {
            "trade_ledger": ledger_path.name,
            "strategy_metrics": metrics_path.name,
            "fold_metrics": folds_path.name,
            "cost_sensitivity": sensitivity_path.name,
            "summary": summary_path.name,
            "manifest": "simulation_manifest.json",
        },
    }
    atomic_write_json(summary_path, summary)
    output_paths = (
        ledger_path,
        metrics_path,
        folds_path,
        sensitivity_path,
        summary_path,
    )
    simulation_manifest = {
        "schema_version": "2.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "audit_directory": audit_dir.name,
            "audit_manifest_sha256": sha256_file(audit_dir / "audit_manifest.json"),
            "audited_predictions_sha256": sha256_file(
                audit_dir / "audited_predictions.csv"
            ),
            "execution_dataset_sha256": audit_summary.get("execution_dataset_sha256"),
        },
        "configuration": {
            "initial_capital": float(initial_capital),
            "notional_per_trade": float(notional_per_trade),
            "cost_scenarios_bps_per_side": list(costs),
            "trading_days_per_year": trading_days_per_year,
            "strategies": list(_STRATEGIES),
        },
        "outputs": manifest_output_entries(output_paths),
    }
    atomic_write_json(simulation_dir / "simulation_manifest.json", simulation_manifest)
    return summary
