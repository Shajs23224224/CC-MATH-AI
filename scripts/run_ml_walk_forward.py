"""Reproducible expanding-window walk-forward evaluation for classical ML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from core.contracts.ml import MLDataset, MLTrainingConfig
from ml import ClassicalMLModel, supported_algorithms


@dataclass(frozen=True)
class WalkForwardFold:
    """One contiguous chronological test window."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def training_samples(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_samples(self) -> int:
        return self.test_end - self.test_start


def build_expanding_folds(
    total_samples: int,
    *,
    n_folds: int = 4,
    test_fraction: float = 0.10,
) -> tuple[WalkForwardFold, ...]:
    """Create consecutive test windows with an expanding training prefix."""
    if isinstance(total_samples, bool) or not isinstance(total_samples, int):
        raise ValueError("total_samples must be an integer")
    if total_samples < 1:
        raise ValueError("total_samples must be positive")

    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 1:
        raise ValueError("n_folds must be a positive integer")

    if (
        isinstance(test_fraction, bool)
        or not math.isfinite(test_fraction)
        or not 0.0 < test_fraction < 0.5
    ):
        raise ValueError("test_fraction must be between 0 and 0.5")

    test_size = math.floor(total_samples * test_fraction)
    if test_size < 2:
        raise ValueError("each test window must contain at least two samples")

    initial_training_size = total_samples - n_folds * test_size
    if initial_training_size < 12:
        raise ValueError("not enough samples for the requested expanding windows")

    folds = []

    for index in range(n_folds):
        test_start = initial_training_size + index * test_size
        test_end = test_start + test_size

        folds.append(
            WalkForwardFold(
                fold=index + 1,
                train_start=0,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )

    if folds[-1].test_end != total_samples:
        raise RuntimeError("walk-forward folds do not cover the final dataset rows")

    for previous, current in zip(folds, folds[1:], strict=False):
        if previous.test_end != current.test_start:
            raise RuntimeError("walk-forward test windows overlap or contain a gap")

    return tuple(folds)


def _training_holdout_fraction(fold: WalkForwardFold) -> float:
    """Represent the exact outer fold using the existing ML chronological split."""
    # The model uses ceil(total_samples * test_fraction).
    # A quarter-sample margin makes the desired integer unambiguous.
    fraction = (fold.test_samples - 0.25) / fold.test_end

    if not 0.05 < fraction < 0.5:
        raise ValueError(
            "fold cannot be represented by MLTrainingConfig test_fraction constraints"
        )

    if math.ceil(fold.test_end * fraction) != fold.test_samples:
        raise RuntimeError("configured holdout does not match the requested fold")

    return fraction


def _prefix_dataset(dataset: MLDataset, end_index: int) -> MLDataset:
    """Build a validated dataset prefix ending at an outer fold boundary."""
    return MLDataset(
        dataset_id=dataset.dataset_id,
        source=dataset.source,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        features=dataset.features[:end_index],
        target=dataset.target[:end_index],
        task=dataset.task,
    )


def _slug(value: object) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return result or "class"


def evaluate_model_fold(
    dataset: MLDataset,
    fold: WalkForwardFold,
    *,
    algorithm: str,
    seed: int,
    dates: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit using only the historical prefix and return OOS predictions."""
    if len(dates) != len(dataset.features):
        raise ValueError("dates must align with dataset rows")

    if (
        fold.train_start != 0
        or fold.train_end != fold.test_start
        or fold.test_end > len(dataset.features)
        or fold.test_samples < 2
    ):
        raise ValueError("invalid expanding-window fold boundaries")

    prefix = _prefix_dataset(dataset, fold.test_end)

    model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm=algorithm,
            task=dataset.task,
            random_seed=seed,
            test_fraction=_training_holdout_fraction(fold),
        )
    )

    evaluation = model.fit(prefix)

    actual_boundaries = (
        evaluation.training_samples,
        evaluation.test_samples,
        evaluation.train_end_index,
        evaluation.test_start_index,
        evaluation.test_end_index,
    )
    expected_boundaries = (
        fold.training_samples,
        fold.test_samples,
        fold.train_end,
        fold.test_start,
        fold.test_end,
    )

    if actual_boundaries != expected_boundaries:
        raise RuntimeError(
            f"{algorithm} did not respect the requested fold boundaries: "
            f"{actual_boundaries} != {expected_boundaries}"
        )

    if evaluation.test_used_for_selection:
        raise RuntimeError("outer test observations were used for model selection")

    test_features = dataset.features[fold.test_start:fold.test_end]
    prediction = model.predict(test_features)

    if len(prediction.predictions) != fold.test_samples:
        raise RuntimeError("prediction count differs from the fold test size")

    record: dict[str, Any] = {
        "fold": fold.fold,
        "task": dataset.task,
        "algorithm": algorithm,
        "status": "COMPLETED",
        "train_start_index": fold.train_start,
        "train_end_index": fold.train_end,
        "test_start_index": fold.test_start,
        "test_end_index": fold.test_end,
        "training_samples": evaluation.training_samples,
        "test_samples": evaluation.test_samples,
        "test_used_for_selection": False,
        "test_start_date": dates[fold.test_start],
        "test_last_feature_date": dates[fold.test_end - 1],
    }
    record.update(evaluation.metrics)

    prediction_rows: list[dict[str, Any]] = []

    for offset, predicted in enumerate(prediction.predictions):
        absolute_index = fold.test_start + offset

        row: dict[str, Any] = {
            "fold": fold.fold,
            "task": dataset.task,
            "algorithm": algorithm,
            "row_index": absolute_index,
            "feature_date": dates[absolute_index],
            "actual": dataset.target[absolute_index],
            "predicted": predicted,
        }

        if prediction.probabilities is not None and prediction.classes is not None:
            for label, probability in zip(
                prediction.classes,
                prediction.probabilities[offset],
                strict=True,
            ):
                row[f"probability_{_slug(label)}"] = probability

        prediction_rows.append(row)

    return record, prediction_rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
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


def _load_snapshot(
    snapshot_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, MLDataset]]:
    """Load and verify the exact snapshot referenced by its manifest."""
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"snapshot not found: {snapshot_path}")

    manifest_path = snapshot_path.parent / "dataset_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifact_info = manifest.get("artifacts", {}).get("dataset_snapshot", {})
    expected_hash = artifact_info.get("sha256")
    expected_filename = artifact_info.get("file")
    actual_hash = _sha256_file(snapshot_path)

    if expected_filename and expected_filename != snapshot_path.name:
        raise ValueError("snapshot filename does not match the source manifest")

    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("snapshot SHA-256 does not match its dataset manifest")

    frame = pd.read_csv(
        snapshot_path,
        index_col="date",
        float_precision="round_trip",
    )

    dates = pd.to_datetime(frame.index, errors="coerce")

    if dates.isna().any():
        raise ValueError("snapshot contains invalid feature dates")

    frame.index = dates

    if frame.index.has_duplicates:
        raise ValueError("snapshot contains duplicate feature dates")

    if not frame.index.is_monotonic_increasing:
        raise ValueError("snapshot is not in chronological order")

    dataset_info = manifest.get("dataset", {})
    feature_names = dataset_info.get("feature_names", [])
    ticker = manifest.get("experiment", {}).get("ticker", "UNKNOWN")

    target_name = "target_next_return"
    direction_name = "target_next_direction"
    required = feature_names + [target_name, direction_name]
    missing = [name for name in required if name not in frame.columns]

    if not feature_names or missing:
        raise ValueError(f"snapshot is missing required columns: {missing}")

    numeric = frame.loc[:, feature_names + [target_name]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    matrix = numeric.loc[:, feature_names].to_numpy(dtype=np.float64)
    regression_target = numeric[target_name].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(regression_target)):
        raise ValueError("snapshot contains non-finite features or targets")

    expected_direction = np.where(regression_target > 0.0, "UP", "DOWN")
    saved_direction = frame[direction_name].astype(str).to_numpy()

    if not np.array_equal(expected_direction, saved_direction):
        raise ValueError("saved classification labels disagree with regression targets")

    expected_rows = dataset_info.get("usable_feature_rows")
    if expected_rows is not None and int(expected_rows) != len(frame):
        raise ValueError("snapshot row count differs from the source manifest")

    features = tuple(
        tuple(float(value) for value in row)
        for row in matrix
    )

    regression_values = tuple(float(value) for value in regression_target)
    classification_values = tuple(str(value) for value in expected_direction)

    datasets = {
        "regression": MLDataset(
            dataset_id=f"{str(ticker).lower()}-walk-forward-regression",
            source="Verified dataset_snapshot.csv; next-session return",
            feature_names=tuple(feature_names),
            target_name=target_name,
            features=features,
            target=regression_values,
            task="regression",
        ),
        "classification": MLDataset(
            dataset_id=f"{str(ticker).lower()}-walk-forward-classification",
            source="Verified dataset_snapshot.csv; next-session direction",
            feature_names=tuple(feature_names),
            target_name=direction_name,
            features=features,
            target=classification_values,
            task="classification",
        ),
    }

    return frame, manifest, datasets


def _aggregate_metrics(
    fold_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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

    for item in fold_records:
        if item["status"] == "COMPLETED":
            grouped[(item["task"], item["algorithm"])].append(item)

    aggregates: list[dict[str, Any]] = []

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
                [item[metric_name] for item in items if metric_name in item],
                dtype=np.float64,
            )

            aggregates.append(
                {
                    "task": task,
                    "algorithm": algorithm,
                    "metric": metric_name,
                    "successful_folds": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

    return aggregates


def run_walk_forward(
    snapshot_path: Path,
    *,
    output_root: Path = Path("artifacts/walk_forward_runs"),
    seed: int = 28,
    n_folds: int = 4,
    test_fraction: float = 0.10,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate all registered classical models sequentially across expanding windows."""
    snapshot_path = snapshot_path.expanduser().resolve()

    frame, source_manifest, datasets = _load_snapshot(snapshot_path)

    folds = build_expanding_folds(
        len(frame),
        n_folds=n_folds,
        test_fraction=test_fraction,
    )

    registry = supported_algorithms()
    total_models = sum(len(algorithms) for algorithms in registry.values())
    total_runs = total_models * len(folds)

    ticker = str(source_manifest.get("experiment", {}).get("ticker", "UNKNOWN"))
    safe_ticker = re.sub(r"[^a-zA-Z0-9_-]+", "_", ticker).lower()

    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"{safe_ticker}_walk_forward_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    date_values = tuple(value.isoformat() for value in frame.index)

    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    completed = 0

    for task in ("regression", "classification"):
        dataset = datasets[task]

        for algorithm in registry[task]:
            for fold in folds:
                completed += 1

                description = (
                    f"[{completed}/{total_runs}] "
                    f"{task}/{algorithm} fold {fold.fold}/{len(folds)}"
                )

                if progress_callback is not None:
                    progress_callback(description)

                try:
                    record, predictions = evaluate_model_fold(
                        dataset,
                        fold,
                        algorithm=algorithm,
                        seed=seed,
                        dates=date_values,
                    )

                    fold_records.append(record)
                    prediction_records.extend(predictions)

                except Exception as exc:
                    failure = {
                        "fold": fold.fold,
                        "task": task,
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

                    fold_records.append(failure)

                    if progress_callback is not None:
                        progress_callback(f"{description} FAILED: {exc}")

    aggregate_records = _aggregate_metrics(fold_records)

    fold_metrics_path = run_dir / "fold_metrics.csv"
    aggregate_path = run_dir / "aggregate_metrics.csv"
    predictions_path = run_dir / "oos_predictions.csv"

    _write_csv(fold_metrics_path, fold_records)
    _write_csv(aggregate_path, aggregate_records)
    _write_csv(predictions_path, prediction_records)

    completed_runs = sum(
        item["status"] == "COMPLETED"
        for item in fold_records
    )
    failed_runs = total_runs - completed_runs

    summary: dict[str, Any] = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "ticker": ticker,
        "status": "COMPLETED" if failed_runs == 0 else "COMPLETED_WITH_FAILURES",
        "seed": seed,
        "snapshot_file": snapshot_path.name,
        "snapshot_sha256": _sha256_file(snapshot_path),
        "total_samples": len(frame),
        "feature_count": len(source_manifest["dataset"]["feature_names"]),
        "fold_count": len(folds),
        "folds": [asdict(fold) for fold in folds],
        "registered_models": total_models,
        "attempted_model_folds": total_runs,
        "completed_model_folds": completed_runs,
        "failed_model_folds": failed_runs,
        "prediction_rows": len(prediction_records),
        "evaluation_scope": (
            "Exploratory historical evaluation. The 2024-2026 holdout "
            "was previously inspected and is not an untouched final validation set."
        ),
        "output_directory": str(run_dir),
        "outputs": {
            "fold_metrics": fold_metrics_path.name,
            "aggregate_metrics": aggregate_path.name,
            "predictions": predictions_path.name,
        },
    }

    summary_path = run_dir / "walk_forward_summary.json"
    _write_json(summary_path, summary)

    output_paths = (
        fold_metrics_path,
        aggregate_path,
        predictions_path,
        summary_path,
    )

    output_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "snapshot_file": snapshot_path.name,
            "snapshot_sha256": _sha256_file(snapshot_path),
            "source_manifest_sha256": _sha256_file(
                snapshot_path.parent / "dataset_manifest.json"
            ),
        },
        "configuration": {
            "seed": seed,
            "fold_count": n_folds,
            "test_fraction_per_fold": test_fraction,
            "algorithms": registry,
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

    _write_json(run_dir / "walk_forward_manifest.json", output_manifest)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible expanding-window walk-forward ML evaluation."
    )

    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/walk_forward_runs"),
    )
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.10)

    args = parser.parse_args()

    def report(message: str) -> None:
        print(message, flush=True)

    summary = run_walk_forward(
        args.snapshot,
        output_root=args.output_root,
        seed=args.seed,
        n_folds=args.folds,
        test_fraction=args.test_fraction,
        progress_callback=report,
    )

    print("\n========== RESUMEN WALK-FORWARD ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if summary["failed_model_folds"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
