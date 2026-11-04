"""Simulate D07.1 audited decisions with explicit execution costs and fixed notional."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_ACTIONS = ("DOWN", "NEUTRAL", "UP")
_REQUIRED_AUDIT_FILES = (
    "audited_predictions.csv",
    "fold_audit.csv",
    "audit_summary.json",
)
_REQUIRED_COLUMNS = (
    "algorithm",
    "fold",
    "row_index",
    "feature_date",
    "entry_date",
    "actual_execution_return",
    "entry_open",
    "exit_close",
    "argmax_prediction",
    "policy_prediction",
    "selected_neutral_threshold",
)
_STRATEGIES = {
    "argmax": "argmax_prediction",
    "neutral_policy": "policy_prediction",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _validate_configuration(
    *,
    initial_capital: float,
    notional_per_trade: float,
    cost_bps_per_side: float,
    trading_days_per_year: int,
) -> None:
    values = {
        "initial_capital": initial_capital,
        "notional_per_trade": notional_per_trade,
        "cost_bps_per_side": cost_bps_per_side,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive")
    if notional_per_trade <= 0.0 or notional_per_trade > initial_capital:
        raise ValueError("notional_per_trade must be positive and <= initial_capital")
    if cost_bps_per_side < 0.0 or cost_bps_per_side > 1000.0:
        raise ValueError("cost_bps_per_side must be between 0 and 1000")
    if (
        isinstance(trading_days_per_year, bool)
        or not isinstance(trading_days_per_year, int)
        or not 1 <= trading_days_per_year <= 366
    ):
        raise ValueError("trading_days_per_year must be an integer between 1 and 366")


def _verify_audit_directory(
    audit_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not audit_dir.is_dir():
        raise FileNotFoundError(f"D07.1 audit directory not found: {audit_dir}")

    manifest_path = audit_dir / "audit_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("D07.1 audit_manifest.json not found")

    manifest = _read_json(manifest_path)
    output_manifest = manifest.get("outputs")
    if not isinstance(output_manifest, dict):
        raise ValueError("D07.1 audit manifest outputs must be an object")

    for filename in _REQUIRED_AUDIT_FILES:
        artifact_path = audit_dir / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"D07.1 artifact is missing: {filename}")

        entry = output_manifest.get(filename)
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        expected_bytes = entry.get("bytes") if isinstance(entry, dict) else None
        actual_hash = _sha256_file(artifact_path)

        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"D07.1 SHA-256 verification failed: {filename}")
        if expected_bytes is not None and artifact_path.stat().st_size != int(expected_bytes):
            raise ValueError(f"D07.1 byte-count verification failed: {filename}")

    summary = _read_json(audit_dir / "audit_summary.json")
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("audit_type") != "D07.1_D06_prediction_integrity"
    ):
        raise ValueError("D07.1 audit summary is not a completed integrity audit")

    checks_passed = int(summary.get("checks_passed", -1))
    checks_total = int(summary.get("checks_total", -1))
    audit_checks = summary.get("checks")
    if (
        checks_passed < 1
        or checks_passed != checks_total
        or not isinstance(audit_checks, list)
        or len(audit_checks) != checks_total
        or any(
            not isinstance(item, dict) or item.get("passed") is not True
            for item in audit_checks
        )
    ):
        raise ValueError("D07.1 audit contains failed or incomplete checks")

    input_manifest = manifest.get("input")
    if not isinstance(input_manifest, dict):
        raise ValueError("D07.1 audit manifest input must be an object")
    if (
        input_manifest.get("decision_policy_run_id")
        != summary.get("decision_policy_run_id")
        or input_manifest.get("execution_dataset_sha256")
        != summary.get("execution_dataset_sha256")
        or input_manifest.get("source_manifest_sha256")
        != summary.get("source_manifest_sha256")
    ):
        raise ValueError("D07.1 manifest inputs do not match its audit summary")

    predictions = pd.read_csv(
        audit_dir / "audited_predictions.csv",
        float_precision="round_trip",
    )
    if len(predictions) != int(summary.get("rows_audited", -1)):
        raise ValueError("D07.1 prediction row count does not match audit summary")

    return predictions, summary, manifest


def _validate_audited_predictions(
    predictions: pd.DataFrame,
    audit_summary: dict[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...], int]:
    missing = set(_REQUIRED_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"D07.1 predictions missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("D07.1 predictions contain no OOS rows")

    algorithms_raw = audit_summary.get("algorithms")
    if (
        not isinstance(algorithms_raw, list)
        or not algorithms_raw
        or any(not isinstance(value, str) or not value for value in algorithms_raw)
        or len(set(algorithms_raw)) != len(algorithms_raw)
    ):
        raise ValueError("D07.1 audit summary has an invalid algorithm list")

    algorithms = tuple(algorithms_raw)
    fold_count = int(audit_summary.get("fold_count", 0))
    if fold_count < 1:
        raise ValueError("D07.1 audit summary has an invalid fold count")

    if set(predictions["algorithm"].astype(str)) != set(algorithms):
        raise ValueError("D07.1 algorithm names do not match its audit summary")

    for column in ("fold", "row_index"):
        values = pd.to_numeric(predictions[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"D07.1 {column} values must be finite integers")
        predictions[column] = values.astype(np.int64)

    for column in ("entry_open", "exit_close", "actual_execution_return"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")

    prices_and_returns = predictions[
        ["entry_open", "exit_close", "actual_execution_return"]
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(prices_and_returns)):
        raise ValueError("D07.1 execution prices/returns contain non-finite values")
    if np.any(prices_and_returns[:, :2] <= 0.0):
        raise ValueError("D07.1 execution prices must be positive")

    calculated_returns = (
        predictions["exit_close"].to_numpy(dtype=np.float64)
        / predictions["entry_open"].to_numpy(dtype=np.float64)
        - 1.0
    )
    observed_returns = predictions["actual_execution_return"].to_numpy(
        dtype=np.float64
    )
    if not np.allclose(
        observed_returns,
        calculated_returns,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("D07.1 execution returns do not match exit_close / entry_open - 1")

    for column in ("feature_date", "entry_date"):
        parsed = pd.to_datetime(predictions[column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError(f"D07.1 {column} contains invalid dates")
        predictions[f"_{column}_parsed"] = parsed

    if (
        predictions["_feature_date_parsed"] >= predictions["_entry_date_parsed"]
    ).any():
        raise ValueError("D07.1 entry dates must follow their feature dates")

    for column in ("argmax_prediction", "policy_prediction"):
        predictions[column] = predictions[column].astype(str)
        if not set(predictions[column]).issubset(EXPECTED_ACTIONS):
            raise ValueError(f"D07.1 {column} contains an unknown action")

    predictions["selected_neutral_threshold"] = pd.to_numeric(
        predictions["selected_neutral_threshold"],
        errors="raise",
    )
    thresholds = predictions["selected_neutral_threshold"].to_numpy(
        dtype=np.float64
    )
    if (
        not np.all(np.isfinite(thresholds))
        or np.any(thresholds < 0.0)
        or np.any(thresholds > 1.0)
    ):
        raise ValueError("D07.1 selected neutral thresholds are invalid")

    if predictions.duplicated(["algorithm", "row_index"]).any():
        raise ValueError("D07.1 predictions contain duplicate algorithm/row_index pairs")

    expected_folds = set(range(1, fold_count + 1))
    canonical_indices: np.ndarray | None = None
    canonical_dates: list[tuple[int, int]] | None = None
    canonical_execution: np.ndarray | None = None

    for algorithm in algorithms:
        selected = predictions.loc[
            predictions["algorithm"].astype(str) == algorithm
        ].sort_values("row_index")
        if selected.empty:
            raise ValueError(f"D07.1 contains no rows for {algorithm}")

        observed_folds = set(selected["fold"].astype(int).tolist())
        if observed_folds != expected_folds:
            raise ValueError(f"D07.1 fold identifiers are incomplete for {algorithm}")

        fold_order = selected.groupby("fold", sort=True)["row_index"].agg(
            ["min", "max"]
        )
        if not fold_order["min"].is_monotonic_increasing:
            raise ValueError(f"D07.1 folds are not chronological for {algorithm}")
        if len(fold_order) > 1:
            previous_max = fold_order["max"].to_numpy(dtype=np.int64)[:-1]
            next_min = fold_order["min"].to_numpy(dtype=np.int64)[1:]
            if np.any(next_min <= previous_max):
                raise ValueError(f"D07.1 fold row ranges overlap for {algorithm}")

        row_indices = selected["row_index"].to_numpy(dtype=np.int64)
        execution_values = selected[
            ["entry_open", "exit_close", "actual_execution_return"]
        ].to_numpy(dtype=np.float64)
        dates = list(
            zip(
                selected["_feature_date_parsed"].astype("int64").tolist(),
                selected["_entry_date_parsed"].astype("int64").tolist(),
                strict=True,
            )
        )
        if canonical_indices is None:
            canonical_indices = row_indices
            canonical_dates = dates
            canonical_execution = execution_values
        else:
            if not np.array_equal(row_indices, canonical_indices):
                raise ValueError("D07.1 algorithms do not share identical OOS row indexes")
            if dates != canonical_dates:
                raise ValueError("D07.1 algorithms do not share identical OOS dates")
            if not np.allclose(
                execution_values,
                canonical_execution,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "D07.1 algorithms do not share identical execution outcomes"
                )

        if not selected["_feature_date_parsed"].is_monotonic_increasing:
            raise ValueError(f"D07.1 feature dates are not chronological for {algorithm}")
        if not selected["_entry_date_parsed"].is_monotonic_increasing:
            raise ValueError(f"D07.1 entry dates are not chronological for {algorithm}")
        if selected["_entry_date_parsed"].duplicated().any():
            raise ValueError(f"D07.1 entry dates contain duplicates for {algorithm}")

        for fold, fold_rows in selected.groupby("fold", sort=True):
            if fold_rows["selected_neutral_threshold"].nunique(dropna=False) != 1:
                raise ValueError(
                    f"D07.1 fold {fold} has inconsistent thresholds for {algorithm}"
                )

    predictions = predictions.sort_values(
        ["algorithm", "fold", "row_index"]
    ).reset_index(drop=True)
    return predictions, algorithms, fold_count


def _simulate_strategy(
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
    bankrupt = False
    peak_equity = float(initial_capital)
    cost_fraction = 2.0 * cost_bps_per_side / 10000.0
    records: list[dict[str, Any]] = []

    for row in ordered.itertuples(index=False):
        requested_action = str(getattr(row, prediction_column))
        requested_direction = (
            1 if requested_action == "UP" else -1 if requested_action == "DOWN" else 0
        )
        equity_before = equity

        if bankrupt:
            executed_direction = 0
            active_notional = 0.0
            gross_pnl = 0.0
            transaction_cost = 0.0
            net_pnl = 0.0
            execution_status = "SKIPPED_BANKRUPTCY"
        elif requested_direction == 0:
            executed_direction = 0
            active_notional = 0.0
            gross_pnl = 0.0
            transaction_cost = 0.0
            net_pnl = 0.0
            execution_status = "NO_TRADE"
        else:
            executed_direction = requested_direction
            active_notional = min(notional_per_trade, equity_before)
            gross_pnl = (
                executed_direction
                * float(row.actual_execution_return)
                * active_notional
            )
            transaction_cost = active_notional * cost_fraction
            net_pnl = gross_pnl - transaction_cost
            execution_status = "EXECUTED"

            equity += net_pnl
            if equity <= 0.0:
                bankrupt = True
                execution_status = "PORTFOLIO_BROKE"

        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0

        records.append(
            {
                "algorithm": algorithm,
                "strategy": strategy,
                "fold": int(row.fold),
                "row_index": int(row.row_index),
                "feature_date": str(row.feature_date),
                "entry_date": str(row.entry_date),
                "requested_action": requested_action,
                "requested_direction": requested_direction,
                "executed_direction": executed_direction,
                "execution_status": execution_status,
                "actual_execution_return": float(row.actual_execution_return),
                "entry_open": float(row.entry_open),
                "exit_close": float(row.exit_close),
                "selected_neutral_threshold": float(
                    row.selected_neutral_threshold
                ),
                "configured_notional": float(notional_per_trade),
                "active_notional": float(active_notional),
                "cost_bps_per_side": float(cost_bps_per_side),
                "gross_return_on_notional": (
                    executed_direction * float(row.actual_execution_return)
                    if active_notional > 0.0
                    else 0.0
                ),
                "gross_pnl": float(gross_pnl),
                "transaction_cost_pnl": float(transaction_cost),
                "net_pnl": float(net_pnl),
                "net_return_on_notional": (
                    float(net_pnl / active_notional)
                    if active_notional > 0.0
                    else 0.0
                ),
                "net_return_on_initial_capital": float(net_pnl / initial_capital),
                "portfolio_equity_before": float(equity_before),
                "portfolio_equity_after": float(equity),
                "drawdown": float(drawdown),
                "bankrupt_after_session": bool(bankrupt),
            }
        )

    return pd.DataFrame.from_records(records)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _strategy_metrics(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
    trading_days_per_year: int,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("cannot calculate metrics for an empty strategy")

    pnl = frame["net_pnl"].to_numpy(dtype=np.float64)
    gross_pnl = frame["gross_pnl"].to_numpy(dtype=np.float64)
    costs = frame["transaction_cost_pnl"].to_numpy(dtype=np.float64)
    directions = frame["executed_direction"].to_numpy(dtype=np.int64)
    traded = directions != 0
    trade_pnl = pnl[traded]
    daily_returns = pnl / initial_capital

    wins = trade_pnl[trade_pnl > 0.0]
    losses = trade_pnl[trade_pnl < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    return_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    downside = np.minimum(daily_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    equity = frame["portfolio_equity_after"].to_numpy(dtype=np.float64)
    equity_with_start = np.concatenate(([initial_capital], equity))
    high_water = np.maximum.accumulate(equity_with_start)[1:]
    drawdowns = np.divide(
        equity,
        high_water,
        out=np.full_like(equity, np.nan, dtype=np.float64),
        where=high_water > 0.0,
    ) - 1.0

    net_pnl = float(pnl.sum())
    final_equity = float(equity[-1])
    sessions = len(frame)
    trades = int(traded.sum())
    total_notional = float(frame["active_notional"].sum())
    skipped_sessions = int(
        (frame["execution_status"] == "SKIPPED_BANKRUPTCY").sum()
    )
    neutral_signals = int((frame["requested_action"] == "NEUTRAL").sum())

    return {
        "sessions": sessions,
        "trades": trades,
        "long_trades": int(np.sum(directions == 1)),
        "short_trades": int(np.sum(directions == -1)),
        "no_trade_sessions": int((frame["execution_status"] == "NO_TRADE").sum()),
        "neutral_signals": neutral_signals,
        "skipped_bankruptcy_sessions": skipped_sessions,
        "exposure_ratio": float(trades / sessions),
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_gross_pnl": float(gross_pnl.sum()),
        "total_transaction_costs": float(costs.sum()),
        "total_net_pnl": net_pnl,
        "simple_total_return_on_initial_capital": float(net_pnl / initial_capital),
        "simple_annualized_return": float(
            (net_pnl / initial_capital) * trading_days_per_year / sessions
        ),
        "annualized_volatility": float(return_std * math.sqrt(trading_days_per_year)),
        "sharpe_ratio": _safe_ratio(
            float(np.mean(daily_returns)) * math.sqrt(trading_days_per_year),
            return_std,
        ),
        "sortino_ratio": _safe_ratio(
            float(np.mean(daily_returns)) * math.sqrt(trading_days_per_year),
            downside_deviation,
        ),
        "maximum_drawdown": (
            float(np.nanmin(drawdowns)) if np.isfinite(drawdowns).any() else None
        ),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "breakeven_trades": int(trades - len(wins) - len(losses)),
        "win_rate": _safe_ratio(float(len(wins)), float(trades)),
        "average_trade_net_pnl": float(np.mean(trade_pnl)) if trades else None,
        "average_winning_trade": float(np.mean(wins)) if len(wins) else None,
        "average_losing_trade": float(np.mean(losses)) if len(losses) else None,
        "profit_factor": _safe_ratio(gross_profit, gross_loss),
        "average_notional_per_trade": _safe_ratio(total_notional, float(trades)),
        "bankrupt": bool(frame["bankrupt_after_session"].any()),
        "bankruptcy_date": (
            str(frame.loc[frame["bankrupt_after_session"], "entry_date"].iloc[0])
            if frame["bankrupt_after_session"].any()
            else None
        ),
        "return_basis": "simple PnL divided by initial capital; no compounding",
    }


def _fold_metrics(
    decisions: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (algorithm, strategy, fold), group in decisions.groupby(
        ["algorithm", "strategy", "fold"],
        sort=True,
    ):
        directions = group["executed_direction"].to_numpy(dtype=np.int64)
        pnl = group["net_pnl"].to_numpy(dtype=np.float64)
        trade_mask = directions != 0
        trade_pnl = pnl[trade_mask]
        records.append(
            {
                "algorithm": str(algorithm),
                "strategy": str(strategy),
                "fold": int(fold),
                "sessions": int(len(group)),
                "trades": int(trade_mask.sum()),
                "long_trades": int(np.sum(directions == 1)),
                "short_trades": int(np.sum(directions == -1)),
                "no_trade_sessions": int(
                    (group["execution_status"] == "NO_TRADE").sum()
                ),
                "neutral_signals": int((group["requested_action"] == "NEUTRAL").sum()),
                "skipped_bankruptcy_sessions": int(
                    (group["execution_status"] == "SKIPPED_BANKRUPTCY").sum()
                ),
                "gross_pnl": float(group["gross_pnl"].sum()),
                "transaction_costs": float(group["transaction_cost_pnl"].sum()),
                "net_pnl": float(pnl.sum()),
                "return_on_initial_capital": float(pnl.sum() / initial_capital),
                "win_rate": _safe_ratio(
                    float(np.sum(trade_pnl > 0.0)),
                    float(len(trade_pnl)),
                ),
                "bankrupt_at_or_before_fold_end": bool(
                    group["bankrupt_after_session"].any()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def simulate_execution_decisions(
    audit_dir: Path,
    *,
    output_root: Path = Path("artifacts/execution_simulations"),
    initial_capital: float = 100000.0,
    notional_per_trade: float = 10000.0,
    cost_bps_per_side: float = 5.0,
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    """Simulate D07.1 decisions without routing orders to a broker.

    Each position is entered at the recorded next-session open and exited at
    the same session's close. A round trip pays twice the per-side cost.
    Notional is fixed per trade, capped at current equity; profits are not
    compounded or reinvested.
    """
    _validate_configuration(
        initial_capital=initial_capital,
        notional_per_trade=notional_per_trade,
        cost_bps_per_side=cost_bps_per_side,
        trading_days_per_year=trading_days_per_year,
    )
    audit_dir = audit_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    predictions, audit_summary, _ = _verify_audit_directory(audit_dir)
    predictions, algorithms, fold_count = _validate_audited_predictions(
        predictions,
        audit_summary,
    )

    all_frames: list[pd.DataFrame] = []
    for algorithm in algorithms:
        algorithm_rows = predictions.loc[
            predictions["algorithm"].astype(str) == algorithm
        ]
        for strategy, prediction_column in _STRATEGIES.items():
            all_frames.append(
                _simulate_strategy(
                    algorithm_rows,
                    algorithm=algorithm,
                    strategy=strategy,
                    prediction_column=prediction_column,
                    initial_capital=initial_capital,
                    notional_per_trade=notional_per_trade,
                    cost_bps_per_side=cost_bps_per_side,
                )
            )

    decisions = pd.concat(all_frames, ignore_index=True)
    decisions = decisions.sort_values(
        ["algorithm", "strategy", "fold", "row_index"]
    ).reset_index(drop=True)

    metric_records: list[dict[str, Any]] = []
    for (algorithm, strategy), group in decisions.groupby(
        ["algorithm", "strategy"],
        sort=True,
    ):
        record: dict[str, Any] = {
            "algorithm": str(algorithm),
            "strategy": str(strategy),
            "cost_bps_per_side": float(cost_bps_per_side),
            "round_trip_cost_bps": float(2.0 * cost_bps_per_side),
            "notional_per_trade": float(notional_per_trade),
            "trading_days_per_year": int(trading_days_per_year),
        }
        record.update(
            _strategy_metrics(
                group,
                initial_capital=initial_capital,
                trading_days_per_year=trading_days_per_year,
            )
        )
        metric_records.append(record)

    metrics = pd.DataFrame.from_records(metric_records)
    fold_metrics = _fold_metrics(
        decisions,
        initial_capital=initial_capital,
    )

    ticker = str(audit_summary.get("ticker", "UNKNOWN")).upper()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    simulation_id = f"{ticker.lower()}_d07_2_{timestamp}"
    simulation_dir = output_root / simulation_id
    simulation_dir.mkdir(parents=False, exist_ok=False)

    decisions_path = simulation_dir / "simulated_decisions.csv"
    metrics_path = simulation_dir / "simulation_metrics.csv"
    fold_metrics_path = simulation_dir / "simulation_fold_metrics.csv"
    summary_path = simulation_dir / "simulation_summary.json"

    _write_csv(decisions_path, decisions)
    _write_csv(metrics_path, metrics)
    _write_csv(fold_metrics_path, fold_metrics)

    cost_model = {
        "cost_bps_per_side": float(cost_bps_per_side),
        "round_trip_cost_bps": float(2.0 * cost_bps_per_side),
        "cost_interpretation": (
            "All-in cost per executed leg; intended to aggregate commissions, "
            "spread and slippage. Applied on entry and exit."
        ),
        "market_impact_and_taxes_included": False,
    }
    execution_model = {
        "entry": "recorded next-session open",
        "exit": "recorded same-session close",
        "holding_period": "one session",
        "actions": {
            "UP": "long",
            "DOWN": "short",
            "NEUTRAL": "flat; no trade and no transaction cost",
        },
        "notional_per_trade": float(notional_per_trade),
        "notional_cap": "min(configured notional, current equity)",
        "compounding": False,
        "reinvestment": False,
        "leverage": False,
        "bankruptcy_rule": (
            "If equity reaches or falls below zero after a trade, no later "
            "positions are opened for that algorithm/strategy."
        ),
        "orders_routed": False,
    }

    summary = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "simulation_type": "D07.2_one_session_decision_simulation",
        "created_at": datetime.now(UTC).isoformat(),
        "simulation_id": simulation_id,
        "ticker": ticker,
        "audit_directory": audit_dir.name,
        "audit_manifest_sha256": _sha256_file(audit_dir / "audit_manifest.json"),
        "audited_predictions_sha256": _sha256_file(
            audit_dir / "audited_predictions.csv"
        ),
        "decision_policy_run_id": audit_summary.get("decision_policy_run_id"),
        "execution_dataset_sha256": audit_summary.get("execution_dataset_sha256"),
        "algorithm_count": len(algorithms),
        "algorithms": list(algorithms),
        "fold_count": fold_count,
        "strategies": list(_STRATEGIES),
        "sessions_per_algorithm": int(
            decisions.groupby("algorithm")["row_index"].nunique().iloc[0]
        ),
        "simulation_rows": len(decisions),
        "capital_model": {
            "initial_capital": float(initial_capital),
            "configured_notional_per_trade": float(notional_per_trade),
            "notional_is_capped_at_current_equity": True,
            "pnl_accounting": "fixed notional; simple additive PnL; no compounding",
        },
        "cost_model": cost_model,
        "execution_model": execution_model,
        "per_strategy": metric_records,
        "limitations": [
            "This is a historical decision simulation, not live trading.",
            "Recorded opening and closing prices are theoretical execution prices.",
            "Costs are user-configured estimates, not broker-verified fees.",
            "Market impact, taxes, borrow fees, rebates, funding and margin rules are not modeled.",
            "Short positions can lose more than their notional; equity can become negative.",
            "The fixed-notional model does not compound or reinvest profits.",
            (
                "The historical OOS folds were already inspected in prior phases; "
                "they are not an untouched prospective test."
            ),
            "Results do not establish future profitability or suitability for capital deployment.",
        ],
        "outputs": {
            "decisions": decisions_path.name,
            "metrics": metrics_path.name,
            "fold_metrics": fold_metrics_path.name,
            "summary": summary_path.name,
            "manifest": "simulation_manifest.json",
        },
    }
    _write_json(summary_path, summary)

    output_paths = (
        decisions_path,
        metrics_path,
        fold_metrics_path,
        summary_path,
    )
    output_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "audit_directory": audit_dir.name,
            "audit_manifest_sha256": _sha256_file(
                audit_dir / "audit_manifest.json"
            ),
            "audited_predictions_sha256": _sha256_file(
                audit_dir / "audited_predictions.csv"
            ),
        },
        "configuration": {
            "initial_capital": float(initial_capital),
            "notional_per_trade": float(notional_per_trade),
            "cost_bps_per_side": float(cost_bps_per_side),
            "trading_days_per_year": int(trading_days_per_year),
            "strategies": list(_STRATEGIES),
        },
        "outputs": {
            path.name: {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        },
    }
    _write_json(simulation_dir / "simulation_manifest.json", output_manifest)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D07.2 simulation of D07.1 audited execution decisions."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_simulations"),
    )
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--notional-per-trade", type=float, default=10000.0)
    parser.add_argument("--cost-bps-per-side", type=float, default=5.0)
    parser.add_argument("--trading-days-per-year", type=int, default=252)
    args = parser.parse_args()

    try:
        summary = simulate_execution_decisions(
            args.audit_dir,
            output_root=args.output_root,
            initial_capital=args.initial_capital,
            notional_per_trade=args.notional_per_trade,
            cost_bps_per_side=args.cost_bps_per_side,
            trading_days_per_year=args.trading_days_per_year,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            f"D07.2 SIMULATION FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    print("\n========== D07.2 SIMULATION SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
