"""D06: nested chronological calibration and evaluation of three-class policies.

A calibration suffix is used only to select the NEUTRAL decision threshold.
It is not probability calibration and the outer OOS test is never used for
threshold, estimator, or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from numbers import Real
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
    FORBIDDEN_FEATURES,
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
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def _write_csv_atomic(records: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(records).to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _slug(value: object) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()
    return result[:32] or "unknown"


def _validate_fraction(value: Real, name: str, lower: float, upper: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not lower < float(value) < upper
    ):
        raise ValueError(f"{name} must be finite and between {lower} and {upper}")
    return float(value)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    return seed


def _holdout_fraction(total_samples: int, holdout_samples: int) -> float:
    """Express an exact integer suffix through MLTrainingConfig's fraction API."""
    if (
        isinstance(total_samples, bool)
        or not isinstance(total_samples, int)
        or isinstance(holdout_samples, bool)
        or not isinstance(holdout_samples, int)
        or not 2 <= holdout_samples < total_samples
    ):
        raise ValueError("invalid chronological holdout size")

    fraction = (holdout_samples - 0.25) / total_samples
    if not 0.05 < fraction < 0.5:
        raise ValueError("holdout fraction violates MLTrainingConfig constraints")
    if math.ceil(total_samples * fraction) != holdout_samples:
        raise RuntimeError("configured holdout does not match expected sample count")
    return fraction


def _prefix_dataset(dataset: MLDataset, end_index: int) -> MLDataset:
    """Return a validated prefix; observations at or after end_index are excluded."""
    if (
        isinstance(end_index, bool)
        or not isinstance(end_index, int)
        or not 12 <= end_index <= len(dataset.features)
    ):
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


def _validate_probability_matrix(
    probabilities: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.shape != (expected_rows, len(EXPECTED_CLASSES)):
        raise ValueError("probability matrix has invalid dimensions")
    if (
        not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("classifier returned invalid probabilities")
    return matrix


def _aligned_probabilities(
    prediction: Any,
    expected_rows: int,
) -> np.ndarray:
    """Align estimator probabilities to the canonical DOWN, NEUTRAL, UP order."""
    if prediction.classes is None or prediction.probabilities is None:
        raise ValueError("classifier must return class probabilities")

    source_classes = tuple(str(value) for value in prediction.classes)
    if (
        len(source_classes) != len(EXPECTED_CLASSES)
        or set(source_classes) != set(EXPECTED_CLASSES)
    ):
        raise ValueError(f"classifier probability labels are invalid: {source_classes}")

    raw = np.asarray(prediction.probabilities, dtype=np.float64)
    if raw.shape != (expected_rows, len(EXPECTED_CLASSES)):
        raise ValueError("probability matrix has invalid dimensions")
    aligned = raw[:, [source_classes.index(label) for label in EXPECTED_CLASSES]]
    return _validate_probability_matrix(aligned, expected_rows)


def apply_neutral_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[str, ...]:
    """Use NEUTRAL when its probability reaches threshold; otherwise UP/DOWN."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(EXPECTED_CLASSES)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("probabilities must be a finite three-column matrix whose rows sum to one")

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("threshold must be finite and between zero and one")

    output: list[str] = []
    for down_probability, neutral_probability, up_probability in matrix:
        if neutral_probability >= float(threshold):
            output.append("NEUTRAL")
        elif down_probability >= up_probability:
            output.append("DOWN")
        else:
            output.append("UP")
    return tuple(output)


def choose_neutral_threshold(
    actual: Sequence[str] | np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Sequence[float] = THRESHOLD_GRID,
) -> dict[str, float]:
    """Select one threshold on a past-only suffix using deterministic tie-breaks.

    The selection score is descriptive and is not an unbiased estimate of
    policy performance because the same suffix selects the threshold.
    """
    y_true = np.asarray(actual, dtype=str)
    matrix = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or len(y_true) == 0 or len(y_true) != len(matrix):
        raise ValueError("calibration labels and probabilities must align")
    if set(y_true.tolist()) != set(EXPECTED_CLASSES):
        raise ValueError("calibration window must contain all three classes")

    matrix = _validate_probability_matrix(matrix, len(y_true))
    if not isinstance(thresholds, Sequence) or isinstance(thresholds, (str, bytes)):
        raise ValueError("threshold grid must be a non-empty sequence")
    if len(thresholds) == 0:
        raise ValueError("threshold grid cannot be empty")

    normalized: list[float] = []
    for threshold in thresholds:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold grid values must be finite and in [0, 1]")
        normalized.append(float(threshold))

    best: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in sorted(set(normalized)):
        predicted = np.asarray(apply_neutral_threshold(matrix, threshold), dtype=str)
        macro_f1 = float(
            f1_score(
                y_true,
                predicted,
                labels=list(EXPECTED_CLASSES),
                average="macro",
                zero_division=0,
            )
        )
        balanced = float(balanced_accuracy_score(y_true, predicted))
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true,
            predicted,
            labels=list(EXPECTED_CLASSES),
            average=None,
            zero_division=0,
        )
        neutral_precision = float(precision[1])
        neutral_recall = float(recall[1])

        key = (macro_f1, balanced, neutral_precision, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": threshold,
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
    predicted: Sequence[str] | np.ndarray,
) -> dict[str, float]:
    """Compute fixed-class metrics and all nine confusion-matrix cells."""
    y_true = np.asarray(actual, dtype=str)
    y_pred = np.asarray(predicted, dtype=str)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("actual and predicted labels must be non-empty aligned vectors")
    if not set(y_true.tolist()).issubset(EXPECTED_CLASSES):
        raise ValueError("actual labels contain an unknown class")
    if not set(y_pred.tolist()).issubset(EXPECTED_CLASSES):
        raise ValueError("predicted labels contain an unknown class")

    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(EXPECTED_CLASSES),
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(EXPECTED_CLASSES))
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
        f"{prefix}_prediction_unique_classes": float(np.unique(y_pred).size),
        f"{prefix}_prediction_collapse": float(np.unique(y_pred).size == 1),
    }

    for index, label in enumerate(EXPECTED_CLASSES):
        predicted_count = int(matrix[:, index].sum())
        metrics[f"{prefix}_{label.lower()}_precision"] = float(precision[index])
        metrics[f"{prefix}_{label.lower()}_recall"] = float(recall[index])
        metrics[f"{prefix}_{label.lower()}_f1"] = float(f1_values[index])
        metrics[f"{prefix}_{label.lower()}_support"] = float(support[index])
        metrics[f"{prefix}_{label.lower()}_predicted_count"] = float(predicted_count)
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
    """Fit, select the policy threshold on a past suffix, and score the outer test."""
    if dataset.task != "classification":
        raise ValueError("decision policy requires a classification dataset")
    if len(dataset.features) != len(dataset.target) or len(dates) != len(dataset.features):
        raise ValueError("features, labels, and dates must align")
    if not algorithm or not isinstance(algorithm, str):
        raise ValueError("algorithm must be a non-empty registered name")
    _validate_seed(seed)
    calibration_fraction = _validate_fraction(
        calibration_fraction,
        "calibration_fraction",
        0.05,
        0.5,
    )

    if (
        fold.train_start != 0
        or fold.train_end != fold.test_start
        or fold.train_end < 12
        or fold.test_end > len(dataset.features)
        or fold.test_samples < 2
        or fold.training_samples != fold.train_end - fold.train_start
    ):
        raise ValueError("invalid expanding fold boundaries")

    outer_train_labels = set(dataset.target[:fold.train_end])
    test_labels = set(dataset.target[fold.test_start:fold.test_end])
    if outer_train_labels != set(EXPECTED_CLASSES):
        raise ValueError("outer training prefix must contain all three classes")
    if test_labels != set(EXPECTED_CLASSES):
        raise ValueError("outer test window must contain all three classes")

    calibration_size = max(2, int(math.ceil(fold.train_end * calibration_fraction)))
    calibration_start = fold.train_end - calibration_size
    if calibration_start < 12:
        raise ValueError("outer training prefix is too short for the calibration suffix")

    calibration_labels = np.asarray(
        dataset.target[calibration_start:fold.train_end],
        dtype=str,
    )
    calibration_training_labels = set(dataset.target[:calibration_start])
    if calibration_training_labels != set(EXPECTED_CLASSES):
        raise ValueError("pre-calibration training prefix must contain all three classes")
    if set(calibration_labels.tolist()) != set(EXPECTED_CLASSES):
        raise ValueError("inner calibration window must contain all three classes")

    # The inner suffix is withheld from estimator fitting and used solely to
    # select the neutral threshold. No outer-test observation is visible here.
    outer_train = _prefix_dataset(dataset, fold.train_end)
    calibration_model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm=algorithm,
            task="classification",
            random_seed=seed,
            test_fraction=_holdout_fraction(fold.train_end, calibration_size),
        )
    )
    calibration_evaluation = calibration_model.fit(outer_train)
    expected_calibration = (
        calibration_start,
        calibration_size,
        calibration_start,
        calibration_start,
        fold.train_end,
    )
    actual_calibration = (
        calibration_evaluation.training_samples,
        calibration_evaluation.test_samples,
        calibration_evaluation.train_end_index,
        calibration_evaluation.test_start_index,
        calibration_evaluation.test_end_index,
    )
    if actual_calibration != expected_calibration or calibration_evaluation.test_used_for_selection:
        raise RuntimeError("inner calibration split does not match the declared boundaries")

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

    # Refit a fresh estimator on the full outer training prefix. The final
    # model's internal holdout mirrors the outer test, but it cannot select
    # thresholds or hyperparameters from that test.
    full_prefix = _prefix_dataset(dataset, fold.test_end)
    final_model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm=algorithm,
            task="classification",
            random_seed=seed,
            test_fraction=_holdout_fraction(fold.test_end, fold.test_samples),
        )
    )
    evaluation = final_model.fit(full_prefix)
    expected_outer = (
        fold.training_samples,
        fold.test_samples,
        fold.train_end,
        fold.test_start,
        fold.test_end,
    )
    actual_outer = (
        evaluation.training_samples,
        evaluation.test_samples,
        evaluation.train_end_index,
        evaluation.test_start_index,
        evaluation.test_end_index,
    )
    if actual_outer != expected_outer or evaluation.test_used_for_selection:
        raise RuntimeError("final model does not respect outer fold boundaries")

    test_prediction = final_model.predict(dataset.features[fold.test_start:fold.test_end])
    probabilities = _aligned_probabilities(test_prediction, fold.test_samples)
    actual = np.asarray(dataset.target[fold.test_start:fold.test_end], dtype=str)

    # Derive argmax explicitly from canonical probability columns so that the
    # persisted decision can be independently reproduced without model objects.
    argmax_indices = np.argmax(probabilities, axis=1)
    argmax_predictions = tuple(EXPECTED_CLASSES[index] for index in argmax_indices)
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
    metrics.update(_policy_metrics("argmax", actual, argmax_predictions))
    metrics.update(_policy_metrics("policy", actual, policy_predictions))

    one_hot = np.column_stack([actual == label for label in EXPECTED_CLASSES]).astype(
        np.float64
    )
    metrics["multiclass_log_loss"] = float(
        log_loss(actual, probabilities, labels=list(EXPECTED_CLASSES))
    )
    metrics["multiclass_brier_score"] = float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )

    training_counts = Counter(dataset.target[:fold.train_end])
    majority_class, majority_count = training_counts.most_common(1)[0]
    majority_predictions = np.full(len(actual), majority_class, dtype=object)
    metrics["majority_baseline_class"] = str(majority_class)
    metrics["majority_baseline_accuracy"] = float(
        accuracy_score(actual, majority_predictions)
    )
    metrics["policy_accuracy_delta_vs_majority"] = (
        metrics["policy_accuracy"] - metrics["majority_baseline_accuracy"]
    )
    metrics["training_majority_share"] = float(majority_count / fold.train_end)

    numeric_metrics = [
        value
        for value in metrics.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not all(math.isfinite(float(value)) for value in numeric_metrics):
        raise RuntimeError("decision policy produced non-finite metrics")

    predictions: list[dict[str, Any]] = []
    for offset in range(fold.test_samples):
        absolute_index = fold.test_start + offset
        predictions.append(
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
    return metrics, predictions


def _aggregate_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        if record.get("status") == "COMPLETED":
            grouped[(str(record["task"]), str(record["algorithm"]))].append(record)

    aggregates: list[dict[str, Any]] = []
    for (task, algorithm), items in sorted(grouped.items()):
        names = sorted(
            {
                key
                for item in items
                for key, value in item.items()
                if key not in excluded
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        for name in names:
            values = np.asarray([item[name] for item in items], dtype=np.float64)
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


def _load_validated_d04_input(
    dataset_path: Path,
) -> tuple[MLDataset, tuple[str, ...], dict[str, Any]]:
    """Validate D04's artifact identity and the execution/target row contract."""
    dataset, dates, manifest = load_execution_dataset(dataset_path)
    dataset_path = dataset_path.expanduser().resolve()
    manifest_path = dataset_path.parent / "execution_dataset_manifest.json"

    experiment = manifest.get("experiment")
    target = manifest.get("target")
    artifact = manifest.get("artifact")
    source = manifest.get("source")
    dataset_info = manifest.get("dataset")
    if not all(isinstance(value, dict) for value in (experiment, target, artifact, source, dataset_info)):
        raise ValueError("D04 manifest is missing required metadata objects")

    threshold_bps = experiment.get("threshold_bps")
    if (
        isinstance(threshold_bps, bool)
        or not isinstance(threshold_bps, Real)
        or not math.isfinite(float(threshold_bps))
        or not 0.0 <= float(threshold_bps) <= 10_000.0
    ):
        raise ValueError("D04 manifest has an invalid threshold_bps")
    if experiment.get("threshold_is_transaction_cost") is not False:
        raise ValueError("D04 threshold must be a target-label band, not transaction cost")
    if target.get("classes") != list(EXPECTED_CLASSES):
        raise ValueError("D04 manifest class order does not match the D06 contract")
    if target.get("classification_name") != "target_execution_direction":
        raise ValueError("D04 manifest classification target is invalid")
    if artifact.get("file") != dataset_path.name:
        raise ValueError("D04 dataset filename does not match its manifest")
    if artifact.get("bytes") is not None and artifact["bytes"] != dataset_path.stat().st_size:
        raise ValueError("D04 dataset byte count does not match its manifest")

    for key in ("raw_ohlcv_sha256", "source_snapshot_sha256", "source_manifest_sha256"):
        if not isinstance(source.get(key), str) or not SHA256_PATTERN.fullmatch(source[key]):
            raise ValueError(f"D04 source identity is missing a valid {key}")
    for key in ("raw_ohlcv_file", "source_snapshot_file", "source_manifest_file"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"D04 source identity is missing {key}")

    feature_names = dataset_info.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or feature_names != list(dataset.feature_names)
        or not feature_names
        or len(set(feature_names)) != len(feature_names)
    ):
        raise ValueError("D04 feature manifest does not match the loaded feature matrix")
    if set(feature_names) & FORBIDDEN_FEATURES:
        raise ValueError("D04 feature manifest contains future execution columns")

    try:
        frame = pd.read_csv(
            dataset_path,
            index_col="feature_date",
            float_precision="round_trip",
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("D04 execution dataset CSV cannot be parsed") from exc
    if frame.empty or len(frame) != len(dataset.target):
        raise ValueError("D04 CSV row count does not match the loaded ML dataset")
    if not frame.columns.is_unique:
        raise ValueError("D04 execution dataset contains duplicate columns")

    required = {
        *feature_names,
        "entry_date",
        "entry_open",
        "exit_close",
        "target_execution_return",
        "target_execution_direction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"D04 execution dataset is missing columns: {sorted(missing)}")

    index = pd.to_datetime(frame.index, errors="coerce")
    if index.isna().any() or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("D04 feature dates are invalid, duplicated, or unordered")
    if tuple(value.isoformat() for value in index) != dates:
        raise ValueError("D04 feature dates do not match the loaded model rows")

    feature_values = frame.loc[:, feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("D04 model features contain non-finite values")

    prices = frame.loc[:, ["entry_open", "exit_close"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        raise ValueError("D04 entry and exit prices must be finite and positive")

    returns = pd.to_numeric(
        frame["target_execution_return"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(returns)):
        raise ValueError("D04 execution returns contain non-finite values")
    calculated_returns = prices[:, 1] / prices[:, 0] - 1.0
    if not np.allclose(returns, calculated_returns, rtol=1e-10, atol=1e-12):
        raise ValueError("D04 execution returns do not match entry/exit prices")

    entry_dates = pd.to_datetime(frame["entry_date"], errors="coerce")
    if entry_dates.isna().any():
        raise ValueError("D04 contains invalid next-session entry dates")
    feature_day_keys = pd.Index(index.strftime("%Y-%m-%d"))
    entry_day_keys = pd.Index(entry_dates.dt.strftime("%Y-%m-%d"))
    if np.any(entry_day_keys <= feature_day_keys):
        raise ValueError("D04 entry dates must be later than their feature dates")
    if entry_day_keys.has_duplicates or not entry_day_keys.is_monotonic_increasing:
        raise ValueError("D04 entry dates must be unique and chronological")

    threshold = float(threshold_bps) / 10_000.0
    expected_labels = np.select(
        [returns > threshold, returns < -threshold],
        ["UP", "DOWN"],
        default="NEUTRAL",
    )
    observed_labels = frame["target_execution_direction"].astype(str).to_numpy()
    if not np.array_equal(observed_labels, expected_labels):
        raise ValueError("D04 class labels do not match the declared threshold rule")
    if tuple(observed_labels.tolist()) != tuple(dataset.target):
        raise ValueError("D04 labels do not match the loaded ML target")

    expected_rows = dataset_info.get("rows")
    if expected_rows != len(frame):
        raise ValueError("D04 manifest row count does not match the dataset")
    if _sha256_file(manifest_path) == "":
        raise RuntimeError("unable to hash D04 manifest")

    return dataset, dates, manifest


def run_decision_policy(
    dataset_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_policy_runs"),
    seed: int = 28,
    n_folds: int = 4,
    test_fraction: float = 0.10,
    calibration_fraction: float = 0.20,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compare argmax to a past-only neutral-threshold policy for each classifier."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    _validate_seed(seed)
    test_fraction = _validate_fraction(test_fraction, "test_fraction", 0.0, 0.5)
    calibration_fraction = _validate_fraction(
        calibration_fraction,
        "calibration_fraction",
        0.05,
        0.5,
    )
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 1:
        raise ValueError("n_folds must be a positive integer")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"D04 execution dataset not found: {dataset_path}")

    dataset, dates, source_manifest = _load_validated_d04_input(dataset_path)
    folds = build_expanding_folds(
        len(dataset.features),
        n_folds=n_folds,
        test_fraction=test_fraction,
    )

    registry = supported_algorithms()["classification"]
    unknown = [algorithm for algorithm in DEFAULT_ALGORITHMS if algorithm not in registry]
    if unknown:
        raise RuntimeError(f"decision-policy algorithms are not registered: {unknown}")

    ticker = str(source_manifest["experiment"].get("ticker", "UNKNOWN"))
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC)
    run_dir = output_root / (
        f"{_slug(ticker)}_decision_policy_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)

    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    total = len(DEFAULT_ALGORITHMS) * len(folds)
    completed = 0

    for algorithm in DEFAULT_ALGORITHMS:
        for fold in folds:
            completed += 1
            message = f"[{completed}/{total}] {algorithm} fold {fold.fold}/{len(folds)}"
            if progress_callback is not None:
                progress_callback(message)
            else:
                print(message, flush=True)

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
                if progress_callback is not None:
                    progress_callback(f"{message} FAILED: {exc}")
                else:
                    print(f"{message} FAILED: {exc}", flush=True)

    fold_path = run_dir / "fold_metrics.csv"
    aggregate_path = run_dir / "aggregate_metrics.csv"
    predictions_path = run_dir / "oos_predictions.csv"
    _write_csv_atomic(fold_records, fold_path)
    _write_csv_atomic(_aggregate_metrics(fold_records), aggregate_path)
    _write_csv_atomic(prediction_records, predictions_path)

    successes = sum(item["status"] == "COMPLETED" for item in fold_records)
    failures = total - successes
    d04_manifest_path = dataset_path.parent / "execution_dataset_manifest.json"
    d04_manifest_sha256 = _sha256_file(d04_manifest_path)

    summary = {
        "run_id": run_dir.name,
        "created_at": timestamp.isoformat(),
        "ticker": ticker,
        "status": "COMPLETED" if failures == 0 else "COMPLETED_WITH_FAILURES",
        "algorithms": list(DEFAULT_ALGORITHMS),
        "fold_count": len(folds),
        "attempted_model_folds": total,
        "completed_model_folds": successes,
        "failed_model_folds": failures,
        "prediction_rows": len(prediction_records),
        "classes": list(EXPECTED_CLASSES),
        "threshold_bps": source_manifest["experiment"]["threshold_bps"],
        "threshold_is_transaction_cost": False,
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_selection_objective": "macro_f1",
        "calibration_fraction": calibration_fraction,
        "outer_test_used_for_selection": False,
        "folds": [
            {
                "fold": fold.fold,
                "train_start_index": fold.train_start,
                "train_end_index": fold.train_end,
                "test_start_index": fold.test_start,
                "test_end_index": fold.test_end,
                "training_samples": fold.training_samples,
                "test_samples": fold.test_samples,
                "test_start_date": dates[fold.test_start],
                "test_last_feature_date": dates[fold.test_end - 1],
            }
            for fold in folds
        ],
        "snapshot_file": dataset_path.name,
        "snapshot_sha256": _sha256_file(dataset_path),
        "source_manifest_sha256": d04_manifest_sha256,
        "d04_original_source_manifest_sha256": source_manifest["source"][
            "source_manifest_sha256"
        ],
        "evaluation_scope": (
            "Exploratory historical nested chronological evaluation. Thresholds "
            "are selected using only a past inner suffix. Previously inspected "
            "historical folds are not an untouched prospective test and do not "
            "establish profitability. The suffix selects a decision threshold; "
            "it does not calibrate the probability estimates."
        ),
        "limitations": [
            "Threshold selection uses the inner suffix; its selection metrics are not unbiased OOS estimates.",
            "Classifier probabilities are model estimates and are not post-hoc calibrated probabilities.",
            "Outer test windows are used only for final descriptive evaluation.",
            "Historical folds may have been inspected during development and are not a prospective test.",
            "D06 does not model transaction costs or execute orders.",
        ],
        "output_directory": str(run_dir),
        "outputs": {
            "fold_metrics": fold_path.name,
            "aggregate_metrics": aggregate_path.name,
            "predictions": predictions_path.name,
        },
    }
    summary_path = run_dir / "decision_policy_summary.json"
    _write_json_atomic(summary_path, summary)

    output_files = (fold_path, aggregate_path, predictions_path, summary_path)
    output_manifest = {
        "schema_version": "1.0",
        "contract_version": "D06.2",
        "created_at": timestamp.isoformat(),
        "input": {
            "execution_dataset": dataset_path.name,
            "execution_dataset_sha256": _sha256_file(dataset_path),
            "source_manifest_sha256": d04_manifest_sha256,
            "d04_original_source_manifest_sha256": source_manifest["source"][
                "source_manifest_sha256"
            ],
            "execution_dataset_bytes": dataset_path.stat().st_size,
            "source_manifest_bytes": d04_manifest_path.stat().st_size,
        },
        "configuration": {
            "seed": seed,
            "fold_count": n_folds,
            "test_fraction": test_fraction,
            "calibration_fraction": calibration_fraction,
            "algorithms": list(DEFAULT_ALGORITHMS),
            "classes": list(EXPECTED_CLASSES),
            "threshold_grid": list(THRESHOLD_GRID),
            "selection_objective": "macro_f1",
            "tie_breaking": [
                "balanced_accuracy",
                "neutral_precision",
                "higher_threshold",
            ],
            "argmax_tie_breaking": "first class in canonical class order",
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
    _write_json_atomic(run_dir / "decision_policy_manifest.json", output_manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nested chronological evaluation of an adaptive NEUTRAL policy."
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
