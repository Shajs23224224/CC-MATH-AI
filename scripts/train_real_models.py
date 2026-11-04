#!/usr/bin/env python3
"""Train CC-MATH-AI's registered classical models on real daily market history.

Downloads daily OHLCV with yfinance, creates causal features and next-session
return/direction labels, runs chronological out-of-sample fit/evaluation, then
persists each fitted sklearn pipeline plus metadata and a JSON summary.

This is a first real-data training run, not proof of profitable predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow execution as `python scripts/train_real_models.py` from a checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from core.contracts.ml import MLDataset, MLTrainingConfig  # noqa: E402
from ml import ClassicalMLModel, supported_algorithms  # noqa: E402

FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "close_vs_ma5",
    "close_vs_ma20",
    "intraday_range_pct",
    "volume_change_1d",
)


def download_history(ticker: str, period: str) -> pd.DataFrame:
    """Download adjusted daily OHLCV; fail clearly rather than train on empty data."""
    frame = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        timeout=25,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no data for {ticker}. Check internet/rate limits, "
            "or rerun using --csv PATH with a local OHLCV CSV."
        )
    if isinstance(frame.columns, pd.MultiIndex):
        # Handles yfinance versions that still return a two-level column index.
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Historical data is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    if len(frame) < 250:
        raise RuntimeError(f"Only {len(frame)} valid daily bars were received; need at least 250.")
    return frame


def read_csv_history(csv_path: Path) -> pd.DataFrame:
    """Read a local OHLCV CSV with a Date column/index and standard OHLCV columns."""
    frame = pd.read_csv(csv_path)
    lower_map = {str(col).strip().lower().replace(" ", "_"): col for col in frame.columns}
    date_col = lower_map.get("date") or lower_map.get("datetime") or lower_map.get("timestamp")
    if date_col is None:
        raise RuntimeError("CSV needs a Date, Datetime, or Timestamp column.")
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    if len(frame) < 250:
        raise RuntimeError(f"CSV contains only {len(frame)} valid rows; need at least 250.")
    return frame


def build_dataset(history: pd.DataFrame, ticker: str) -> tuple[MLDataset, MLDataset, pd.DataFrame]:
    """Build features known at close t and labels for the next session t+1."""
    close = history["close"].astype(float)
    daily_return = close.pct_change()
    features = pd.DataFrame(index=history.index)
    features["return_1d"] = daily_return
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["volatility_5d"] = daily_return.rolling(5, min_periods=5).std()
    features["volatility_10d"] = daily_return.rolling(10, min_periods=10).std()
    features["volatility_20d"] = daily_return.rolling(20, min_periods=20).std()
    features["close_vs_ma5"] = close / close.rolling(5, min_periods=5).mean() - 1.0
    features["close_vs_ma20"] = close / close.rolling(20, min_periods=20).mean() - 1.0
    features["intraday_range_pct"] = (
        history["high"].astype(float) - history["low"].astype(float)
    ) / close.replace(0.0, np.nan)
    features["volume_change_1d"] = history["volume"].astype(float).pct_change()

    next_return = close.shift(-1) / close - 1.0
    dataset_frame = features.copy()
    dataset_frame["target_next_return"] = next_return
    # Only the next-session label is used, never future values in the features.
    dataset_frame = dataset_frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(dataset_frame) < 250:
        raise RuntimeError(
            f"Only {len(dataset_frame)} rows remain after feature construction; need at least 250."
        )

    x = tuple(
        tuple(float(value) for value in row)
        for row in dataset_frame.loc[:, FEATURE_COLUMNS].to_numpy()
    )
    regression_target = tuple(float(v) for v in dataset_frame["target_next_return"].to_numpy())
    direction = tuple("UP" if value > 0 else "DOWN" for value in regression_target)
    ds_id = f"{ticker.lower()}-daily-next-session-v1"
    regression_dataset = MLDataset(
        dataset_id=ds_id,
        source=(
            f"Yahoo Finance adjusted daily OHLCV for {ticker}; "
            "causal features; next-session return"
        ),
        feature_names=FEATURE_COLUMNS,
        target_name="next_session_return",
        features=x,
        target=regression_target,
        task="regression",
    )
    classification_dataset = MLDataset(
        dataset_id=f"{ds_id}-direction",
        source=(
            f"Yahoo Finance adjusted daily OHLCV for {ticker}; "
            "causal features; next-session direction"
        ),
        feature_names=FEATURE_COLUMNS,
        target_name="next_session_direction",
        features=x,
        target=direction,
        task="classification",
    )
    return regression_dataset, classification_dataset, dataset_frame


def json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol; default SPY (S&P 500 ETF).")
    parser.add_argument("--period", default="10y", help="History period, e.g. 5y, 10y, max.")
    parser.add_argument("--csv", type=Path, help="Optional local OHLCV CSV instead of downloading.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "training_runs",
        help="Where fitted model artifacts and reports will be written.",
    )
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()
    history = read_csv_history(args.csv) if args.csv else download_history(ticker, args.period)
    regression_ds, classification_ds, rows = build_dataset(history, ticker)

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"{ticker.lower()}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "source": "local_csv" if args.csv else "Yahoo Finance adjusted daily OHLCV",
                "period_requested": None if args.csv else args.period,
                "downloaded_rows": len(history),
                "usable_feature_rows": len(rows),
                "first_usable_date": str(rows.index[0]),
                "last_usable_date": str(rows.index[-1]),
                "feature_names": list(FEATURE_COLUMNS),
                "target_regression": "next_session_return = close[t+1]/close[t] - 1",
                "target_classification": "UP if next_session_return > 0 else DOWN",
                "warning": (
                    "Historical predictive fit does not establish profitability "
                    "or future performance."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    registry = supported_algorithms()
    jobs = [("regression", name, regression_ds) for name in registry["regression"]]
    jobs += [("classification", name, classification_ds) for name in registry["classification"]]
    run_results: list[dict[str, Any]] = []

    print(f"Ticker: {ticker}; barras: {len(history)}; filas entrenables: {len(rows)}")
    print(f"Rango: {rows.index[0]} -> {rows.index[-1]}")
    print(f"Modelos registrados: {len(jobs)}. Salida: {run_dir}")

    for index, (task, algorithm, dataset) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Entrenando {task}/{algorithm} ...", flush=True)
        model = ClassicalMLModel(
            MLTrainingConfig(
                algorithm=algorithm,
                task=task,
                random_seed=args.seed,
                test_fraction=0.2,
            )
        )
        try:
            evaluation = model.fit(dataset)
            if model._pipeline is None or model.metadata is None:
                raise RuntimeError("Training returned without a fitted pipeline and metadata.")
            model_key = f"{task}__{algorithm}"
            model_path = run_dir / f"{model_key}.joblib"
            # Persist the actual fitted scikit-learn pipeline, not only its metadata.
            joblib.dump(model._pipeline, model_path, compress=3)
            record = {
                "task": task,
                "algorithm": algorithm,
                "status": "TRAINED",
                "artifact": model_path.name,
                "metadata": json_safe(model.metadata),
                "evaluation": json_safe(evaluation),
            }
            run_results.append(record)
            print(f"    OK metrics={evaluation.metrics}")
        except Exception as exc:  # continue through registry; preserve exact failure per model
            record = {
                "task": task,
                "algorithm": algorithm,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            run_results.append(record)
            print(f"    FALLÓ {type(exc).__name__}: {exc}")

    summary = {
        "run_id": run_stamp,
        "ticker": ticker,
        "random_seed": args.seed,
        "total_models": len(jobs),
        "trained_models": sum(item["status"] == "TRAINED" for item in run_results),
        "failed_models": sum(item["status"] == "FAILED" for item in run_results),
        "results": run_results,
        "interpretation": (
            "Models were fitted on a chronological training prefix and evaluated "
            "on a held-out tail. This is a first real-data baseline, not proof "
            "of trading profitability."
        ),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    metrics_rows = []
    for item in run_results:
        evaluation = item.get("evaluation", {})
        metrics_rows.append(
            {
                "task": item["task"],
                "algorithm": item["algorithm"],
                "status": item["status"],
                **evaluation.get("metrics", {}),
                "artifact": item.get("artifact", ""),
                "error": item.get("error", ""),
            }
        )
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    print("\n=== RESUMEN ===")
    print(f"Entrenados: {summary['trained_models']}/{summary['total_models']}")
    print(f"Fallidos:   {summary['failed_models']}")
    print(f"Artefactos e informes: {run_dir}")
    print(
        "Nota: comparar modelos solo con el tramo de prueba no basta para "
        "decidir una estrategia de inversión."
    )
    return 0 if summary["trained_models"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
