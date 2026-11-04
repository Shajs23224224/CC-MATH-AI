"""Nested chronological evaluation of adaptive three-class decision policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from core.contracts.ml import MLDataset, MLTrainingConfig
from ml import ClassicalMLModel, supported_algorithms
from scripts.run_execution_ml_walk_forward import (
    EXPECTED_CLASSES,
    load_execution_dataset,
)
from scripts.run_ml_walk_forward import (
    WalkForwardFold,
    build_expanding_folds,
)

DEFAULT_ALGORITHMS = (
    "logistic_regression",
    "gradient_boosting_classifier",
    "random_forest_classifier",
)

THRESHOLD_GRID = tuple(
    float(value)
    for value in np.round(np.arange(0.05, 0.801, 0.01), 2)
)


def _holdout_fraction(total_samples: int, holdout_samples: int) -> float:
    """Represent an exact chronological holdout with MLTrainingConfig."""
    if not 2 <= holdout_samples < total_samples:
        raise ValueError("invalid chronological holdout size")

    fraction = (holdout_samples - 0.25) / total_samples

    if not 0.05 < fraction < 0.5:
        raise ValueError("holdout fraction violates MLTrainingConfig constraints")

    if math.ceil(total_samples * fraction) != holdout_samples:
        raise RuntimeError("configured holdout does not match expected sample count")

    return fraction


def _prefix_dataset(dataset: MLDataset, end_index: int) -> MLDataset:
    """Return a validated chronological prefix without future observations."""
    if not 12 <= end_index <= len(dataset.features):
        raise ValueError("invalid dataset prefix boundary")

    return MLDataset(
        dataset_id=dataset.dataset_id,
        source=dataset.source,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        features=dataset.features[:end_index],
        target=dataset.target[:end_index],
        task=dataset.task,
    )


def _aligned_probabilities(
    prediction: Any,
    expected_rows: int,
) -> np.ndarray:
    """Normalize sklearn probabilities to DOWN, NEUTRAL, UP column order."""
    if prediction.classes is None or prediction.probabilities is None:
        raise ValueError("classifier must return class probabilities")

    source_classes = tuple(str(value) for value in prediction.classes)

    if set(source_classes) != set(EXPECTED_CLASSES):
        raise ValueError(
            f"classifier probability labels are invalid: {source_classes}"
        )

    indices = [
        source_classes.index(label)
        for label in EXPECTED_CLASSES
    ]

    probabilities = np.asarray(
        prediction.probabilities,
        dtype=np.float64,
    )

    if probabilities.shape != (expected_rows, len(EXPECTED_CLASSES)):
        raise ValueError("probability matrix has invalid dimensions")

    probabilities = probabilities[:, indices]

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("classifier returned invalid probabilities")

    return probabilities


def apply_neutral_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[str, ...]:
    """Select NEUTRAL at threshold; otherwise choose between UP and DOWN."""
    matrix = np.asarray(probabilities, dtype=np.float64)

    if (
        matrix.ndim != 2
        or matrix.shape[1] != 3
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise ValueError("probabilities must be a finite three-column matrix")

    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")

    if isinstance(threshold, bool) or not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")

    output = []

    for down_probability, neutral_probability, up_probability in matrix:
        if neutral_probability >= threshold:
            output.append("NEUTRAL")
        elif down_probability >= up_probability:
            output.append("DOWN")
        else:
            output.append("UP")

    return tuple(output)


def choose_neutral_threshold(
    actual: tuple[str, ...] | list[str] | np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: tuple[float, ...] = THRESHOLD_GRID,
) -> dict[str, float]:
    """Choose a neutral threshold on a past-only calibration window.

    The primary objective is macro F1. Ties are resolved by balanced accuracy,
    neutral precision and then the more conservative (higher) threshold.
    """
    y_true = np.asarray(actual, dtype=str)
    matrix = np.asarray(probabilities, dtype=np.float64)

    if len(y_true) != len(matrix) or len(y_true) == 0:
        raise ValueError("calibration labels and probabilities must align")

    if set(y_true.tolist()) != set(EXPECTED_CLASSES):
        raise ValueError("calibration window must contain all three classes")

    if not thresholds:
        raise ValueError("threshold grid cannot be empty")

    best: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None

    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold grid values must be in [0, 1]")

        predicted = np.asarray(
            apply_neutral_threshold(matrix, threshold),
            dtype=str,
        )

        macro_f1 = float(
            f1_score(
                y_true,
                predicted,
                labels=list(EXPECTED_CLASSES),
                average="macro",
                zero_division=0,
            )
        )
        balanced = float(
            balanced_accuracy_score(y_true, predicted)
        )

        precision, recall, _, _ = precision_recall_fscore_support(
            y_true,
            predicted,
            labels=list(EXPECTED_CLASSES),
            average=None,
            zero_division=0,
        )

        neutral_precision = float(precision[1])
        neutral_recall = float(recall[1])

        # Prefer macro F1 first. Deterministic tie-breaking is performed
        # only on the calibration window, never on the outer test window.
        key = (
            macro_f1,
            balanced,
            neutral_precision,
            float(threshold),
        )

        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced,
                "neutral_precision": neutral_precision,
                "neutral_recall": neutral_recall,
            }

    if best is None:
        raise RuntimeError("no valid neutral threshold could be selected")

    return best


def _policy_metrics(
    prefix: str,
    actual: np.ndarray,
    predicted: tuple[str, ...] | np.ndarray,
) -> dict[str, float]:
    """Compute fixed-label multiclass metrics and confusion counts."""
    y_true = np.asarray(actual, dtype=str)
    y_pred = np.asarray(predicted, dtype=str)

    if len(y_true) != len(y_pred):
        raise ValueError("actual and predicted labels do not align")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(EXPECTED_CLASSES),
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(EXPECTED_CLASSES),
    )

    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        f"{prefix}_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(EXPECTED_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        f"{prefix}_prediction_unique_classes": float(
            np.unique(y_pred).size
        ),
        f"{prefix}_prediction_collapse": float(
            np.unique(y_pred).size == 1
        ),
    }

    for index, label in enumerate(EXPECTED_CLASSES):
        predicted_count = int(matrix[:, index].sum())

        metrics[f"{prefix}_{label.lower()}_precision"] = float(
            precision[index]
        )
        metrics[f"{prefix}_{label.lower()}_recall"] = float(
            recall[index]
        )
        metrics[f"{prefix}_{label.lower()}_f1"] = float(f1[index])
        metrics[f"{prefix}_{label.lower()}_support"] = float(support[index])
        metrics[f"{prefix}_{label.lower()}_predicted_count"] = float(
            predicted_count
        )
        metrics[f"{prefix}_{label.lower()}_predicted_share"] = float(
            predicted_count / len(y_true)
        )

    for actual_index, actual_label in enumerate(EXPECTED_CLASSES):
        for predicted_index, predicted_label in enumerate(EXPECTED_CLASSES):
            metrics[
                f"{prefix}_confusion_{actual_label.lower()}_"
                f"predicted_{predicted_label.lower()}"
            ] = float(matrix[actual_index, predicted_index])

    return metrics


def evaluate_decision_policy_fold(
    dataset: MLDataset,
    fold: WalkForwardFold,
    *,
    algorithm: str,
    seed: int,
    dates: tuple[str, ...],
    calibration_fraction: float = 0.20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calibrate on an inner validation suffix and evaluate once on outer test."""
    if dataset.task != "classification":
        raise ValueError("decision policy requires a classification dataset")

    if len(dates) != len(dataset.features):
        raise ValueError("dates must align with dataset rows")

    if (
        fold.train_start != 0
        or fold.train_end != fold.test_start
        or fold.test_end > len(dataset.features)
    ):
        raise ValueError("invalid expanding fold boundaries")

    if not 0.05 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0.05 and 0.5")

    train_labels = set(dataset.target[:fold.train_end])
    test_labels = set(dataset.target[fold.test_start:fold.test_end])

    if train_labels != set(EXPECTED_CLASSES):
        raise ValueError("outer training prefix must contain all three classes")

    if test_labels != set(EXPECTED_CLASSES):
        raise ValueError("outer test window must contain all three classes")

    # STEP 1: create a single chronological calibration suffix inside
    # the outer training prefix.
    outer_train = _prefix_dataset(dataset, fold.train_end)

    calibration_size = max(
        2,
        int(math.ceil(fold.train_end * calibration_fraction)),
    )
    calibration_start = fold.train_end - calibration_size

    calibration_labels = np.asarray(
        dataset.target[calibration_start:fold.train_end],
        dtype=str,
    )

    if set(calibration_labels.tolist()) != set(EXPECTED_CLASSES):
        raise ValueError("inner calibration window must contain all three classes")

    calibration_model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm=algorithm,
            task="classification",
            random_seed=seed,
            test_fraction=_holdout_fraction(
                fold.train_end,
                calibration_size,
            ),
        )
    )

    calibration_evaluation = calibration_model.fit(outer_train)

    if (
        calibration_evaluation.training_samples != calibration_start
        or calibration_evaluation.test_samples != calibration_size
        or calibration_evaluation.train_end_index != calibration_start
        or calibration_evaluation.test_start_index != calibration_start
        or calibration_evaluation.test_end_index != fold.train_end
        or calibration_evaluation.test_used_for_selection
    ):
        raise RuntimeError("inner calibration split does not match expected boundaries")

    calibration_prediction = calibration_model.predict(
        dataset.features[calibration_start:fold.train_end]
    )

    calibration_probabilities = _aligned_probabilities(
        calibration_prediction,
        calibration_size,
    )

    selected = choose_neutral_threshold(
        calibration_labels,
        calibration_probabilities,
    )

    # STEP 2: refit a fresh model using the complete outer training prefix.
    # The outer test suffix is excluded from all threshold and hyperparameter
    # selection. It is used only by the final evaluation model.
    full_prefix = _prefix_dataset(dataset, fold.test_end)

    final_model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm=algorithm,
            task="classification",
            random_seed=seed,
            test_fraction=_holdout_fraction(
                fold.test_end,
                fold.test_samples,
            ),
        )
    )

    evaluation = final_model.fit(full_prefix)

    if (
        evaluation.training_samples != fold.training_samples
        or evaluation.test_samples != fold.test_samples
        or evaluation.train_end_index != fold.train_end
        or evaluation.test_start_index != fold.test_start
        or evaluation.test_end_index != fold.test_end
        or evaluation.test_used_for_selection
    ):
        raise RuntimeError("final model does not respect outer fold boundaries")

    test_prediction = final_model.predict(
        dataset.features[fold.test_start:fold.test_end]
    )

    probabilities = _aligned_probabilities(
        test_prediction,
        fold.test_samples,
    )

    actual = np.asarray(
        dataset.target[fold.test_start:fold.test_end],
        dtype=str,
    )

    argmax_predictions = tuple(
        str(value)
        for value in test_prediction.predictions
    )

    policy_predictions = apply_neutral_threshold(
        probabilities,
        selected["threshold"],
    )

    metrics: dict[str, Any] = {
        "fold": fold.fold,
        "task": "classification",
        "algorithm": algorithm,
        "status": "COMPLETED",
        "train_start_index": fold.train_start,
        "train_end_index": fold.train_end,
        "test_start_index": fold.test_start,
        "test_end_index": fold.test_end,
        "training_samples": fold.training_samples,
        "test_samples": fold.test_samples,
        "test_used_for_selection": False,
        "test_start_date": dates[fold.test_start],
        "test_last_feature_date": dates[fold.test_end - 1],
        "calibration_start_index": calibration_start,
        "calibration_end_index": fold.train_end,
        "calibration_samples": calibration_size,
        "calibration_test_used_for_selection": True,
        "threshold_objective": "macro_f1",
        "selected_neutral_threshold": selected["threshold"],
        "calibration_macro_f1": selected["macro_f1"],
        "calibration_balanced_accuracy": selected["balanced_accuracy"],
        "calibration_neutral_precision": selected["neutral_precision"],
        "calibration_neutral_recall": selected["neutral_recall"],
    }

    metrics.update(
        _policy_metrics("argmax", actual, argmax_predictions)
    )
    metrics.update(
        _policy_metrics("policy", actual, policy_predictions)
    )

    one_hot = np.column_stack(
        [actual == label for label in EXPECTED_CLASSES]
    ).astype(np.float64)

    metrics["multiclass_log_loss"] = float(
        log_loss(
            actual,
            probabilities,
            labels=list(EXPECTED_CLASSES),
        )
    )
    metrics["multiclass_brier_score"] = float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )

    training_counts = Counter(dataset.target[:fold.train_end])
    majority_class, majority_count = training_counts.most_common(1)[0]
    majority_predictions = np.full(
        len(actual),
        majority_class,
        dtype=object,
    )

    metrics["majority_baseline_class"] = str(majority_class)
    metrics["majority_baseline_accuracy"] = float(
        accuracy_score(actual, majority_predictions)
    )
    metrics["policy_accuracy_delta_vs_majority"] = (
        metrics["policy_accuracy"]
        - metrics["majority_baseline_accuracy"]
    )
    metrics["training_majority_share"] = float(
        majority_count / fold.train_end
    )

    prediction_rows: list[dict[str, Any]] = []

    for offset in range(fold.test_samples):
        absolute_index = fold.test_start + offset

        prediction_rows.append(
            {
                "fold": fold.fold,
                "algorithm": algorithm,
                "row_index": absolute_index,
                "feature_date": dates[absolute_index],
                "actual": actual[offset],
                "argmax_prediction": argmax_predictions[offset],
                "policy_prediction": policy_predictions[offset],
                "selected_neutral_threshold": selected["threshold"],
                "probability_down": float(probabilities[offset, 0]),
                "probability_neutral": float(probabilities[offset, 1]),
                "probability_up": float(probabilities[offset, 2]),
            }
        )

    numeric_metrics = [
        value
        for value in metrics.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]

    if not all(math.isfinite(float(value)) for value in numeric_metrics):
        raise RuntimeError("decision policy produced non-finite metrics")

    return metrics, prediction_rows


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


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")

    pd.DataFrame(records).to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    temporary.replace(path)


def _aggregate_metrics(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
        "calibration_start_index",
        "calibration_end_index",
        "calibration_samples",
        "calibration_test_used_for_selection",
        "threshold_objective",
        "majority_baseline_class",
    }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if record["status"] == "COMPLETED":
            grouped[(record["task"], record["algorithm"])].append(record)

    aggregates: list[dict[str, Any]] = []

    for (task, algorithm), items in sorted(grouped.items()):
        metric_names = sorted(
            {
                key
                for item in items
                for key, value in item.items()
                if key not in excluded
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )

        for name in metric_names:
            values = np.asarray(
                [item[name] for item in items],
                dtype=np.float64,
            )

            aggregates.append(
                {
                    "task": task,
                    "algorithm": algorithm,
                    "metric": name,
                    "successful_folds": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

    return aggregates


def run_decision_policy(
    dataset_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_policy_runs"),
    seed: int = 28,
    n_folds: int = 4,
    test_fraction: float = 0.10,
    calibration_fraction: float = 0.20,
) -> dict[str, Any]:
    """Compare argmax with a past-only calibrated NEUTRAL threshold."""
    dataset, dates, source_manifest = load_execution_dataset(
        dataset_path.expanduser().resolve()
    )

    folds = build_expanding_folds(
        len(dataset.features),
        n_folds=n_folds,
        test_fraction=test_fraction,
    )

    registry = supported_algorithms()["classification"]

    unknown = [
        algorithm
        for algorithm in DEFAULT_ALGORITHMS
        if algorithm not in registry
    ]

    if unknown:
        raise RuntimeError(f"decision-policy algorithms are not registered: {unknown}")

    ticker = str(
        source_manifest.get("experiment", {}).get("ticker", "UNKNOWN")
    )

    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"{ticker.lower()}_decision_policy_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    total = len(DEFAULT_ALGORITHMS) * len(folds)
    completed = 0

    for algorithm in DEFAULT_ALGORITHMS:
        for fold in folds:
            completed += 1
            print(
                f"[{completed}/{total}] {algorithm} fold "
                f"{fold.fold}/{len(folds)}",
                flush=True,
            )

            try:
                record, predictions = evaluate_decision_policy_fold(
                    dataset,
                    fold,
                    algorithm=algorithm,
                    seed=seed,
                    dates=dates,
                    calibration_fraction=calibration_fraction,
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
                print(f"FAILED: {exc}", flush=True)

    fold_path = run_dir / "fold_metrics.csv"
    aggregate_path = run_dir / "aggregate_metrics.csv"
    predictions_path = run_dir / "oos_predictions.csv"

    _write_csv(fold_path, fold_records)
    _write_csv(aggregate_path, _aggregate_metrics(fold_records))
    _write_csv(predictions_path, prediction_records)

    successes = sum(
        item["status"] == "COMPLETED"
        for item in fold_records
    )
    failures = total - successes

    summary = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "ticker": ticker,
        "status": "COMPLETED" if failures == 0 else "COMPLETED_WITH_FAILURES",
        "algorithms": list(DEFAULT_ALGORITHMS),
        "fold_count": len(folds),
        "attempted_model_folds": total,
        "completed_model_folds": successes,
        "failed_model_folds": failures,
        "prediction_rows": len(prediction_records),
        "classes": list(EXPECTED_CLASSES),
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_selection_objective": "macro_f1",
        "calibration_fraction": calibration_fraction,
        "outer_test_used_for_selection": False,
        "snapshot_file": Path(dataset_path).name,
        "snapshot_sha256": _sha256_file(Path(dataset_path)),
        "evaluation_scope": (
            "Exploratory historical nested chronological evaluation. "
            "Thresholds are selected using only an inner past validation "
            "suffix. Previously inspected historical folds are not an "
            "untouched prospective test and do not establish profitability."
        ),
        "output_directory": str(run_dir),
        "outputs": {
            "fold_metrics": fold_path.name,
            "aggregate_metrics": aggregate_path.name,
            "predictions": predictions_path.name,
        },
    }

    summary_path = run_dir / "decision_policy_summary.json"
    _write_json(summary_path, summary)

    output_files = (
        fold_path,
        aggregate_path,
        predictions_path,
        summary_path,
    )

    output_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "execution_dataset": Path(dataset_path).name,
            "execution_dataset_sha256": _sha256_file(Path(dataset_path)),
            "source_manifest_sha256": _sha256_file(
                Path(dataset_path).parent / "execution_dataset_manifest.json"
            ),
        },
        "configuration": {
            "seed": seed,
            "fold_count": n_folds,
            "test_fraction": test_fraction,
            "calibration_fraction": calibration_fraction,
            "algorithms": list(DEFAULT_ALGORITHMS),
            "threshold_grid": list(THRESHOLD_GRID),
            "selection_objective": "macro_f1",
            "tie_breaking": [
                "balanced_accuracy",
                "neutral_precision",
                "higher_threshold",
            ],
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
            for path in output_files
        },
    }

    _write_json(run_dir / "decision_policy_manifest.json", output_manifest)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nested temporal evaluation of an adaptive NEUTRAL decision policy."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_policy_runs"),
    )
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)

    args = parser.parse_args()

    summary = run_decision_policy(
        args.dataset,
        output_root=args.output_root,
        seed=args.seed,
        n_folds=args.folds,
        test_fraction=args.test_fraction,
        calibration_fraction=args.calibration_fraction,
    )

    print("\n========== D06 RESUMEN ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if summary["failed_model_folds"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
