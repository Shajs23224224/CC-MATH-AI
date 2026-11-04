"""Registered real-market training tasks for the local dashboard."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yfinance as yf

from apps.dashboard import TrainingCancelledError, TrainingOrchestrator
from core.contracts.ml import MLTrainingConfig
from core.utils.market_dataset_artifacts import persist_market_dataset_artifacts
from core.contracts.training import (
    TrainingProgress,
    TrainingProgressCallback,
)
from ml import ClassicalMLModel, supported_algorithms
from scripts.train_real_models import (
    FEATURE_COLUMNS,
    build_dataset,
    download_history,
    json_safe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training_runs"

TASK_NAME = "real_market.train_classical_suite"

_ALLOWED_PERIODS = {"2y", "5y", "10y", "max"}
_ALLOWED_PARAMETERS = {"ticker", "period", "seed"}


def _validate_parameters(parameters: Mapping[str, object]) -> tuple[str, str, int]:
    """Accept only bounded, explicitly supported training parameters."""
    unknown = set(parameters) - _ALLOWED_PARAMETERS
    if unknown:
        raise ValueError(f"Unsupported training parameters: {', '.join(sorted(unknown))}")

    ticker_value = parameters.get("ticker", "SPY")
    period_value = parameters.get("period", "10y")
    seed_value = parameters.get("seed", 28)

    if not isinstance(ticker_value, str):
        raise ValueError("ticker must be a string")
    ticker = ticker_value.strip().upper()

    if not re.fullmatch(r"[A-Z0-9.^=_-]{1,24}", ticker):
        raise ValueError("ticker contains unsupported characters")

    if not isinstance(period_value, str):
        raise ValueError("period must be a string")
    period = period_value.strip().lower()

    if period not in _ALLOWED_PERIODS:
        raise ValueError(f"period must be one of: {', '.join(sorted(_ALLOWED_PERIODS))}")

    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed_value <= 2**32 - 1:
        raise ValueError("seed is outside the supported range")

    return ticker, period, seed_value


def _overall_progress(model_index: int, model_count: int, local_percent: float) -> float:
    """Map one model's progress into the entire training-suite progress range."""
    bounded = min(100.0, max(0.0, local_percent))
    return 8.0 + 90.0 * ((model_index - 1) + bounded / 100.0) / model_count



def _sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a persisted artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically without silently accepting NaN or infinity."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_snapshot(frame: pd.DataFrame, path: Path) -> str:
    """Persist a DataFrame with round-trip-safe floating-point precision."""
    temporary = path.with_name(path.name + ".tmp")

    frame.to_csv(
        temporary,
        index=True,
        index_label="date",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
        lineterminator="\n",
    )

    temporary.replace(path)
    return _sha256_file(path)


def _persist_reproducibility_artifacts(
    run_dir: Path,
    history: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    ticker: str,
    period: str,
    seed: int,
) -> dict[str, Any]:
    """Use the shared reproducibility contract used by the standalone CLI."""
    return persist_market_dataset_artifacts(
        run_dir,
        history,
        rows,
        ticker=ticker,
        source="Yahoo Finance adjusted daily OHLCV",
        period_requested=period,
        seed=seed,
        feature_names=FEATURE_COLUMNS,
        provider_version=getattr(yf, "__version__", "unknown"),
    )


def _write_outputs(
    run_dir: Path,
    *,
    run_id: str,
    ticker: str,
    seed: int,
    jobs: list[tuple[str, str, object]],
    results: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    """Persist the suite summary and per-model metrics."""
    trained = sum(item["status"] == "TRAINED" for item in results)
    failed = sum(item["status"] == "FAILED" for item in results)

    summary = {
        "run_id": run_id,
        "ticker": ticker,
        "random_seed": seed,
        "total_models": len(jobs),
        "trained_models": trained,
        "failed_models": failed,
        "status": status,
        "results": results,
        "interpretation": (
            "Models were fitted on a chronological training prefix and evaluated "
            "on a held-out tail. This is a real-data research baseline, not proof "
            "of trading profitability or future predictive performance."
        ),
    }

    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    metrics_rows = []
    for item in results:
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
    return summary


def train_real_market_classical_suite(
    parameters: Mapping[str, object],
    progress_callback: TrainingProgressCallback,
    cancellation,
) -> dict[str, Any]:
    """Train every registered classical model on one real historical dataset."""
    ticker, period, seed = _validate_parameters(parameters)

    cancellation.raise_if_cancelled()
    progress_callback(
        TrainingProgress(
            stage="data_download",
            progress_percent=2.0,
            message=f"Downloading daily market history for {ticker}",
            attributes={"ticker": ticker, "period": period},
        )
    )

    history = download_history(ticker, period)
    cancellation.raise_if_cancelled()

    progress_callback(
        TrainingProgress(
            stage="dataset_construction",
            progress_percent=6.0,
            metrics={"downloaded_rows": float(len(history))},
            message="Constructing causal features and next-session targets",
        )
    )

    regression_ds, classification_ds, rows = build_dataset(history, ticker)
    cancellation.raise_if_cancelled()

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = DEFAULT_OUTPUT_DIR / f"{ticker.lower()}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset_manifest = _persist_reproducibility_artifacts(
        run_dir,
        history,
        rows,
        ticker=ticker,
        period=period,
        seed=seed,
    )
    cancellation.raise_if_cancelled()

    progress_callback(
        TrainingProgress(
            stage="dataset_snapshot_persisted",
            progress_percent=7.0,
            metrics={
                "raw_rows": float(len(history)),
                "snapshot_rows": float(len(rows)),
            },
            attributes={
                "ticker": ticker,
                "raw_ohlcv_sha256": dataset_manifest["artifacts"]["raw_ohlcv"]["sha256"],
                "dataset_snapshot_sha256": dataset_manifest["artifacts"][
                    "dataset_snapshot"
                ]["sha256"],
                "manifest_file": "dataset_manifest.json",
            },
            message="Exact source and derived dataset snapshots persisted and hashed",
        )
    )

    dataset_summary = {
        "ticker": ticker,
        "source": "Yahoo Finance adjusted daily OHLCV",
        "period_requested": period,
        "downloaded_rows": len(history),
        "usable_feature_rows": len(rows),
        "raw_ohlcv_file": "raw_ohlcv.csv",
        "dataset_snapshot_file": "dataset_snapshot.csv",
        "dataset_manifest_file": "dataset_manifest.json",
        "raw_ohlcv_sha256": dataset_manifest["artifacts"]["raw_ohlcv"]["sha256"],
        "dataset_snapshot_sha256": dataset_manifest["artifacts"]["dataset_snapshot"]["sha256"],
        "direction_distribution": dataset_manifest["dataset"]["direction_distribution"],
        "first_usable_date": str(rows.index[0]),
        "last_usable_date": str(rows.index[-1]),
        "feature_names": list(FEATURE_COLUMNS),
        "target_regression": "next_session_return = close[t+1]/close[t] - 1",
        "target_classification": "UP if next_session_return > 0 else DOWN",
        "warning": (
            "Historical predictive fit does not establish profitability "
            "or future performance."
        ),
    }

    (run_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    registry = supported_algorithms()
    jobs: list[tuple[str, str, object]] = [
        ("regression", name, regression_ds) for name in registry["regression"]
    ]
    jobs.extend(
        ("classification", name, classification_ds)
        for name in registry["classification"]
    )

    results: list[dict[str, Any]] = []
    model_count = len(jobs)

    progress_callback(
        TrainingProgress(
            stage="training_prepared",
            progress_percent=8.0,
            metrics={
                "downloaded_rows": float(len(history)),
                "usable_feature_rows": float(len(rows)),
                "registered_models": float(model_count),
            },
            attributes={"ticker": ticker, "output_dir": str(run_dir)},
            message=f"Dataset ready; {model_count} registered models will be trained",
        )
    )

    try:
        for index, (task, algorithm, dataset) in enumerate(jobs, start=1):
            cancellation.raise_if_cancelled()

            progress_callback(
                TrainingProgress(
                    stage="model_start",
                    step=index - 1,
                    total_steps=model_count,
                    progress_percent=_overall_progress(index, model_count, 0.0),
                    attributes={
                        "task": task,
                        "algorithm": algorithm,
                        "model_index": index,
                        "total_models": model_count,
                    },
                    message=f"Starting {task}/{algorithm}",
                )
            )

            model = ClassicalMLModel(
                MLTrainingConfig(
                    algorithm=algorithm,
                    task=task,
                    random_seed=seed,
                    test_fraction=0.2,
                )
            )

            def model_progress(
                update: TrainingProgress,
                *,
                current_index: int = index,
                current_task: str = task,
                current_algorithm: str = algorithm,
            ) -> None:
                cancellation.raise_if_cancelled()

                local_percent = update.progress_percent
                if local_percent is None:
                    if update.step is not None and update.total_steps:
                        local_percent = 100.0 * update.step / update.total_steps
                    else:
                        local_percent = 50.0

                namespaced_metrics = {
                    f"{current_task}.{current_algorithm}.{key}": value
                    for key, value in update.metrics.items()
                }

                attributes = dict(update.attributes)
                attributes.update(
                    {
                        "task": current_task,
                        "algorithm": current_algorithm,
                        "model_index": current_index,
                        "total_models": model_count,
                    }
                )

                progress_callback(
                    TrainingProgress(
                        stage=update.stage,
                        step=update.step,
                        total_steps=update.total_steps,
                        progress_percent=_overall_progress(
                            current_index, model_count, local_percent
                        ),
                        metrics=namespaced_metrics,
                        attributes=attributes,
                        message=update.message,
                    )
                )

            try:
                evaluation = model.fit(
                    dataset,
                    progress_callback=model_progress,
                )
                cancellation.raise_if_cancelled()

                if model._pipeline is None or model.metadata is None:
                    raise RuntimeError(
                        "Training returned without a fitted pipeline and metadata."
                    )

                model_key = f"{task}__{algorithm}"
                model_path = run_dir / f"{model_key}.joblib"
                joblib.dump(model._pipeline, model_path, compress=3)

                record = {
                    "task": task,
                    "algorithm": algorithm,
                    "status": "TRAINED",
                    "artifact": model_path.name,
                    "metadata": json_safe(model.metadata),
                    "evaluation": json_safe(evaluation),
                }
                results.append(record)

                print(
                    f"[{index}/{model_count}] OK {task}/{algorithm}: "
                    f"{evaluation.metrics}",
                    flush=True,
                )

            except TrainingCancelledError:
                raise
            except Exception as exc:
                results.append(
                    {
                        "task": task,
                        "algorithm": algorithm,
                        "status": "FAILED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"[{index}/{model_count}] FAILED {task}/{algorithm}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            cancellation.raise_if_cancelled()

            progress_callback(
                TrainingProgress(
                    stage="model_finished",
                    step=index,
                    total_steps=model_count,
                    progress_percent=_overall_progress(index, model_count, 100.0),
                    metrics={
                        "models_trained": float(
                            sum(item["status"] == "TRAINED" for item in results)
                        ),
                        "models_failed": float(
                            sum(item["status"] == "FAILED" for item in results)
                        ),
                    },
                    attributes={
                        "task": task,
                        "algorithm": algorithm,
                        "model_index": index,
                    },
                    message=f"Finished {task}/{algorithm}",
                )
            )

    except TrainingCancelledError:
        _write_outputs(
            run_dir,
            run_id=run_stamp,
            ticker=ticker,
            seed=seed,
            jobs=jobs,
            results=results,
            status="CANCELLED_PARTIAL",
        )
        raise

    summary = _write_outputs(
        run_dir,
        run_id=run_stamp,
        ticker=ticker,
        seed=seed,
        jobs=jobs,
        results=results,
        status=(
            "COMPLETED_WITH_ERRORS"
            if any(item["status"] == "FAILED" for item in results)
            else "COMPLETED"
        ),
    )

    if summary["trained_models"] == 0:
        raise RuntimeError(
            f"No models were trained successfully. See {run_dir / 'training_summary.json'}"
        )

    progress_callback(
        TrainingProgress(
            stage="suite_completed",
            step=model_count,
            total_steps=model_count,
            progress_percent=100.0,
            metrics={
                "models_trained": float(summary["trained_models"]),
                "models_failed": float(summary["failed_models"]),
            },
            attributes={
                "ticker": ticker,
                "output_dir": str(run_dir),
            },
            message=(
                f"Training suite completed: {summary['trained_models']} trained, "
                f"{summary['failed_models']} failed"
            ),
        )
    )

    print(f"Training artifacts: {run_dir}", flush=True)
    return summary


def register_training_tasks(orchestrator: TrainingOrchestrator) -> None:
    """Register the allow-listed real-market classical training task."""
    orchestrator.register_task(
        TASK_NAME,
        train_real_market_classical_suite,
        description=(
            "Train all registered classical regression and classification models "
            "using real historical daily market data."
        ),
    )


__all__ = [
    "TASK_NAME",
    "register_training_tasks",
    "train_real_market_classical_suite",
]
