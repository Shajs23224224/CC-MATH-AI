"""Three-class expanding-window evaluation for execution-aware ML datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import log_loss

from core.contracts.ml import MLDataset
from ml import supported_algorithms
from scripts.run_ml_walk_forward import (
    WalkForwardFold,
    build_expanding_folds,
    evaluate_model_fold,
)

EXPECTED_CLASSES = ("DOWN", "NEUTRAL", "UP")

FORBIDDEN_FEATURES = {
    "entry_date",
    "entry_open",
    "exit_close",
    "target_execution_return",
    "target_execution_direction",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")

    pd.DataFrame(rows).to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    temporary.replace(path)


def load_execution_dataset(
    dataset_path: Path,
) -> tuple[MLDataset, tuple[str, ...], dict[str, Any]]:
    """Load a verified execution dataset without exposing future-price columns."""
    dataset_path = dataset_path.expanduser().resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"execution dataset not found: {dataset_path}")

    manifest_path = dataset_path.parent / "execution_dataset_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError("execution dataset manifest not found")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifact = manifest.get("artifact", {})
    if artifact.get("file") != dataset_path.name:
        raise ValueError("dataset filename does not match its manifest")

    expected_hash = artifact.get("sha256")
    actual_hash = _sha256_file(dataset_path)

    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("execution dataset SHA-256 verification failed")

    frame = pd.read_csv(
        dataset_path,
        index_col="feature_date",
        float_precision="round_trip",
    )

    dates = pd.to_datetime(frame.index, errors="coerce")

    if dates.isna().any():
        raise ValueError("execution dataset contains invalid feature dates")

    frame.index = dates

    if frame.index.has_duplicates:
        raise ValueError("execution dataset contains duplicate feature dates")

    if not frame.index.is_monotonic_increasing:
        raise ValueError("execution dataset is not chronologically ordered")

    dataset_info = manifest.get("dataset", {})
    feature_names = dataset_info.get("feature_names", [])
    target_name = manifest.get("target", {}).get(
        "classification_name",
        "target_execution_direction",
    )

    if not feature_names:
        raise ValueError("execution manifest does not define feature_names")

    leaked_features = FORBIDDEN_FEATURES.intersection(feature_names)

    if leaked_features:
        raise ValueError(
            f"future execution columns cannot be model features: {sorted(leaked_features)}"
        )

    missing = [
        name for name in list(feature_names) + [target_name]
        if name not in frame.columns
    ]

    if missing:
        raise ValueError(f"execution dataset is missing columns: {missing}")

    features_frame = frame.loc[:, feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    )

    matrix = features_frame.to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("execution features contain non-finite values")

    labels = frame[target_name].astype(str).to_numpy()
    observed_classes = set(labels.tolist())

    if observed_classes != set(EXPECTED_CLASSES):
        raise ValueError(
            f"expected exactly three classes {EXPECTED_CLASSES}; "
            f"found {sorted(observed_classes)}"
        )

    expected_rows = dataset_info.get("rows")
    if expected_rows is not None and int(expected_rows) != len(frame):
        raise ValueError("dataset row count does not match its manifest")

    dataset = MLDataset(
        dataset_id=(
            f"{manifest.get('experiment', {}).get('ticker', 'UNKNOWN').lower()}"
            "-execution-direction-3class"
        ),
        source=(
            "Verified execution_dataset.csv; next-session open-to-close "
            "three-class direction"
        ),
        feature_names=tuple(feature_names),
        target_name=target_name,
        features=tuple(
            tuple(float(value) for value in row)
            for row in matrix
        ),
        target=tuple(str(label) for label in labels),
        task="classification",
    )

    date_values = tuple(value.isoformat() for value in frame.index)

    return dataset, date_values, manifest


def evaluate_execution_fold(
    dataset: MLDataset,
    fold: WalkForwardFold,
    *,
    algorithm: str,
    seed: int,
    dates: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one three-class classifier and score its probability estimates."""
    train_labels = set(dataset.target[:fold.train_end])
    test_labels = set(dataset.target[fold.test_start:fold.test_end])

    if train_labels != set(EXPECTED_CLASSES):
        raise ValueError(
            f"fold {fold.fold} training prefix does not contain all three classes"
        )

    if test_labels != set(EXPECTED_CLASSES):
        raise ValueError(
            f"fold {fold.fold} test window does not contain all three classes"
        )

    record, predictions = evaluate_model_fold(
        dataset,
        fold,
        algorithm=algorithm,
        seed=seed,
        dates=dates,
    )

    probability_columns = [
        f"probability_{label.lower()}"
        for label in EXPECTED_CLASSES
    ]

    if not predictions or any(
        column not in predictions[0]
        for column in probability_columns
    ):
        raise RuntimeError(
            f"{algorithm} did not provide all three class probabilities"
        )

    probabilities = np.asarray(
        [
            [row[column] for column in probability_columns]
            for row in predictions
        ],
        dtype=np.float64,
    )

    if (
        probabilities.shape != (fold.test_samples, len(EXPECTED_CLASSES))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise RuntimeError("classifier returned invalid three-class probabilities")

    actual = np.asarray(
        dataset.target[fold.test_start:fold.test_end],
        dtype=str,
    )

    one_hot = np.column_stack(
        [actual == label for label in EXPECTED_CLASSES]
    ).astype(np.float64)

    record["multiclass_log_loss"] = float(
        log_loss(
            actual,
            probabilities,
            labels=list(EXPECTED_CLASSES),
        )
    )
    record["multiclass_brier_score"] = float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )

    return record, predictions


def _aggregate_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    excluded = {
        "fold",
        "task",
        "algorithm",
        "status",
        "train_start_index",
        "train_end_index",
        "test_start_index",
        "test_end_index",
        "training_samples",
        "test_samples",
        "test_used_for_selection",
        "test_start_date",
        "test_last_feature_date",
    }

    for record in records:
        if record["status"] == "COMPLETED":
            grouped[(record["task"], record["algorithm"])].append(record)

    output: list[dict[str, Any]] = []

    for (task, algorithm), items in sorted(grouped.items()):
        metric_names = sorted(
            {
                key
                for item in items
                for key, value in item.items()
                if key not in excluded and isinstance(value, (int, float))
            }
        )

        for metric_name in metric_names:
            values = np.asarray(
                [item[metric_name] for item in items],
                dtype=np.float64,
            )

            output.append(
                {
                    "task": task,
                    "algorithm": algorithm,
                    "metric": metric_name,
                    "successful_folds": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

    return output


def run_execution_walk_forward(
    dataset_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_ml_runs"),
    seed: int = 28,
    n_folds: int = 4,
    test_fraction: float = 0.10,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run all six registered classifiers through four expanding time windows."""
    dataset_path = dataset_path.expanduser().resolve()

    dataset, dates, source_manifest = load_execution_dataset(dataset_path)

    folds = build_expanding_folds(
        len(dataset.features),
        n_folds=n_folds,
        test_fraction=test_fraction,
    )

    registry = supported_algorithms()["classification"]

    total_runs = len(registry) * len(folds)

    ticker = str(
        source_manifest.get("experiment", {}).get("ticker", "UNKNOWN")
    )
    safe_ticker = re.sub(r"[^a-zA-Z0-9_-]+", "_", ticker).lower()

    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"{safe_ticker}_three_class_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    completed = 0

    for algorithm in registry:
        for fold in folds:
            completed += 1
            description = (
                f"[{completed}/{total_runs}] "
                f"{algorithm} fold {fold.fold}/{len(folds)}"
            )

            if progress_callback is not None:
                progress_callback(description)

            try:
                record, predictions = evaluate_execution_fold(
                    dataset,
                    fold,
                    algorithm=algorithm,
                    seed=seed,
                    dates=dates,
                )
                fold_records.append(record)
                prediction_records.extend(predictions)

            except Exception as exc:
                fold_records.append(
                    {
                        "fold": fold.fold,
                        "task": "classification",
                        "algorithm": algorithm,
                        "status": "FAILED",
                        "train_start_index": fold.train_start,
                        "train_end_index": fold.train_end,
                        "test_start_index": fold.test_start,
                        "test_end_index": fold.test_end,
                        "training_samples": fold.training_samples,
                        "test_samples": fold.test_samples,
                        "test_used_for_selection": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                )

                if progress_callback is not None:
                    progress_callback(f"{description} FAILED: {exc}")

    aggregate_records = _aggregate_metrics(fold_records)

    fold_path = run_dir / "fold_metrics.csv"
    aggregate_path = run_dir / "aggregate_metrics.csv"
    predictions_path = run_dir / "oos_predictions.csv"

    _write_csv(fold_path, fold_records)
    _write_csv(aggregate_path, aggregate_records)
    _write_csv(predictions_path, prediction_records)

    completed_runs = sum(
        record["status"] == "COMPLETED"
        for record in fold_records
    )
    failed_runs = total_runs - completed_runs

    summary = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "ticker": ticker,
        "status": "COMPLETED" if failed_runs == 0 else "COMPLETED_WITH_FAILURES",
        "task": "three_class_execution_direction",
        "classes": list(EXPECTED_CLASSES),
        "threshold_bps": source_manifest["experiment"]["threshold_bps"],
        "threshold_is_transaction_cost": False,
        "features": list(dataset.feature_names),
        "seed": seed,
        "fold_count": len(folds),
        "folds": [asdict(fold) for fold in folds],
        "registered_models": len(registry),
        "attempted_model_folds": total_runs,
        "completed_model_folds": completed_runs,
        "failed_model_folds": failed_runs,
        "prediction_rows": len(prediction_records),
        "snapshot_file": dataset_path.name,
        "snapshot_sha256": _sha256_file(dataset_path),
        "evaluation_scope": (
            "Exploratory historical evaluation. This is not an untouched "
            "prospective test and does not establish trading profitability."
        ),
        "output_directory": str(run_dir),
        "outputs": {
            "fold_metrics": fold_path.name,
            "aggregate_metrics": aggregate_path.name,
            "predictions": predictions_path.name,
        },
    }

    summary_path = run_dir / "execution_ml_summary.json"
    _write_json(summary_path, summary)

    output_paths = (
        fold_path,
        aggregate_path,
        predictions_path,
        summary_path,
    )

    output_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "execution_dataset": dataset_path.name,
            "execution_dataset_sha256": _sha256_file(dataset_path),
            "source_manifest_sha256": _sha256_file(
                dataset_path.parent / "execution_dataset_manifest.json"
            ),
        },
        "configuration": {
            "seed": seed,
            "fold_count": n_folds,
            "test_fraction": test_fraction,
            "algorithms": list(registry),
            "classes": list(EXPECTED_CLASSES),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {
            path.name: {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        },
    }

    _write_json(run_dir / "execution_ml_manifest.json", output_manifest)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-class execution-aware expanding-window ML evaluation."
    )

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_ml_runs"),
    )
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.10)

    args = parser.parse_args()

    def report(message: str) -> None:
        print(message, flush=True)

    summary = run_execution_walk_forward(
        args.dataset,
        output_root=args.output_root,
        seed=args.seed,
        n_folds=args.folds,
        test_fraction=args.test_fraction,
        progress_callback=report,
    )

    print("\n========== D05 RESUMEN ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if summary["failed_model_folds"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
