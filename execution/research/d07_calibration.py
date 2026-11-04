"""D07.3: independently reproduce D06.3 temperature calibration and reliability."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from execution.research.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    manifest_output_entries,
    read_json_object,
    sha256_file,
    verify_manifest_outputs,
)
from scripts.run_execution_ml_walk_forward import EXPECTED_CLASSES, load_execution_dataset
from scripts.run_ml_walk_forward import build_expanding_folds

_D06_OUTPUTS = (
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "oos_predictions.csv",
    "decision_policy_summary.json",
    "calibration_predictions.csv",
)
_RAW_COLUMNS = (
    "raw_probability_down",
    "raw_probability_neutral",
    "raw_probability_up",
)
_CALIBRATED_COLUMNS = (
    "probability_down",
    "probability_neutral",
    "probability_up",
)
_CALIBRATION_METRICS = (
    "probability_calibration_temperature",
    "selected_neutral_threshold",
    "calibration_raw_log_loss",
    "calibration_scaled_log_loss",
    "calibration_raw_brier_score",
    "calibration_scaled_brier_score",
    "calibration_macro_f1",
    "calibration_balanced_accuracy",
    "calibration_neutral_precision",
    "calibration_neutral_recall",
    "multiclass_log_loss",
    "multiclass_brier_score",
    "multiclass_expected_calibration_error",
    "multiclass_maximum_calibration_error",
    "down_expected_calibration_error",
    "down_maximum_calibration_error",
    "neutral_expected_calibration_error",
    "neutral_maximum_calibration_error",
    "up_expected_calibration_error",
    "up_maximum_calibration_error",
)
_BIN_KEY = ("algorithm", "fold", "class", "bin")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_integer(value: object, field: str) -> int:
    _require(
        not isinstance(value, (bool, np.bool_)),
        f"{field} must be an integer",
    )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    _require(
        np.isfinite(numeric) and numeric.is_integer(),
        f"{field} must be an integer",
    )
    return int(numeric)


def _as_finite(value: object, field: str) -> float:
    _require(
        not isinstance(value, (bool, np.bool_)),
        f"{field} must be numeric, not boolean",
    )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    _require(np.isfinite(result), f"{field} must be finite")
    return result


def _close(observed: object, expected: float, field: str) -> None:
    value = _as_finite(observed, field)
    _require(
        bool(np.isclose(value, expected, rtol=1e-7, atol=1e-9)),
        f"{field} mismatch: expected {expected:.12g}, received {value:.12g}",
    )


def _probability_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, str, str],
    field: str,
) -> np.ndarray:
    _require(set(columns).issubset(frame.columns), f"{field} probability schema is incomplete")
    matrix = frame.loc[:, list(columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    _require(
        matrix.ndim == 2
        and matrix.shape[1] == len(EXPECTED_CLASSES)
        and np.all(np.isfinite(matrix))
        and np.all(matrix >= 0.0)
        and np.all(matrix <= 1.0)
        and np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6),
        f"{field} probabilities are not finite three-class simplex rows",
    )
    return matrix


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    _require(
        matrix.ndim == 2
        and matrix.shape[1] == len(EXPECTED_CLASSES)
        and np.all(np.isfinite(matrix))
        and np.all((matrix >= 0.0) & (matrix <= 1.0))
        and np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6),
        "temperature input probabilities are invalid",
    )
    _require(
        np.isfinite(temperature) and temperature > 0.0,
        "temperature must be finite and positive",
    )
    logits = np.log(np.clip(matrix, 1e-15, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _log_loss(actual: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(actual, dtype=str)
    matrix = np.asarray(probabilities, dtype=np.float64)
    _require(
        labels.ndim == 1
        and len(labels) == len(matrix)
        and len(labels) > 0
        and set(labels.tolist()) == set(EXPECTED_CLASSES),
        "calibration labels must contain all three canonical classes",
    )
    indices = np.asarray(
        [EXPECTED_CLASSES.index(label) for label in labels],
        dtype=np.int64,
    )
    selected = matrix[np.arange(len(labels)), indices]
    return float(-np.mean(np.log(np.clip(selected, 1e-15, 1.0))))


def _brier_score(actual: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(actual, dtype=str)
    one_hot = np.column_stack([labels == label for label in EXPECTED_CLASSES]).astype(
        np.float64
    )
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _fit_temperature_independently(
    actual: np.ndarray,
    raw_probabilities: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, dict[str, float]]:
    labels = np.asarray(actual, dtype=str)
    raw = np.asarray(raw_probabilities, dtype=np.float64)
    _require(
        labels.ndim == 1
        and len(labels) == len(raw)
        and set(labels.tolist()) == set(EXPECTED_CLASSES),
        "calibration suffix must contain all three classes",
    )
    lower = math.log(bounds[0])
    upper = math.log(bounds[1])

    def objective(log_temperature: float) -> float:
        scaled = _apply_temperature(raw, math.exp(log_temperature))
        return _log_loss(labels, scaled)

    optimized = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-8},
    )
    _require(
        bool(optimized.success) and np.isfinite(float(optimized.fun)),
        "independent temperature optimization did not converge",
    )
    candidates = (lower, upper, 0.0, float(optimized.x))
    best_log_temperature = min(
        candidates,
        key=lambda value: (objective(value), abs(value)),
    )
    temperature = float(math.exp(best_log_temperature))
    scaled = _apply_temperature(raw, temperature)
    return temperature, {
        "calibration_raw_log_loss": _log_loss(labels, raw),
        "calibration_scaled_log_loss": _log_loss(labels, scaled),
        "calibration_raw_brier_score": _brier_score(labels, raw),
        "calibration_scaled_brier_score": _brier_score(labels, scaled),
    }


def _select_threshold_independently(
    actual: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    labels = np.asarray(actual, dtype=str)
    best: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in sorted(set(thresholds)):
        predicted = _neutral_decisions(probabilities, threshold)
        macro_f1 = float(
            f1_score(
                labels,
                predicted,
                labels=list(EXPECTED_CLASSES),
                average="macro",
                zero_division=0,
            )
        )
        balanced = float(balanced_accuracy_score(labels, predicted))
        precision, recall, _, _ = precision_recall_fscore_support(
            labels,
            predicted,
            labels=list(EXPECTED_CLASSES),
            average=None,
            zero_division=0,
        )
        neutral_precision = float(precision[1])
        neutral_recall = float(recall[1])
        key = (macro_f1, balanced, neutral_precision, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "selected_neutral_threshold": float(threshold),
                "calibration_macro_f1": macro_f1,
                "calibration_balanced_accuracy": balanced,
                "calibration_neutral_precision": neutral_precision,
                "calibration_neutral_recall": neutral_recall,
            }
    _require(best is not None, "D06 threshold grid produced no candidate")
    return best


def _neutral_decisions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    output: list[str] = []
    for down_probability, neutral_probability, up_probability in probabilities:
        if neutral_probability >= threshold:
            output.append("NEUTRAL")
        elif down_probability >= up_probability:
            output.append("DOWN")
        else:
            output.append("UP")
    return np.asarray(output, dtype=str)


def _reliability_diagnostics(
    actual: np.ndarray,
    probabilities: np.ndarray,
    *,
    algorithm: str,
    fold: int,
    n_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    labels = np.asarray(actual, dtype=str)
    _require(
        labels.ndim == 1
        and len(labels) == len(probabilities)
        and len(labels) > 0
        and set(labels.tolist()) == set(EXPECTED_CLASSES),
        "OOS reliability sample must contain all three classes",
    )
    rows: list[dict[str, Any]] = []
    class_ece: list[float] = []
    class_mce: list[float] = []
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        confidence = probabilities[:, class_index]
        observed = (labels == class_name).astype(np.float64)
        bin_indices = np.minimum(
            np.floor(confidence * n_bins).astype(np.int64),
            n_bins - 1,
        )
        gaps: list[float] = []
        weighted_error = 0.0
        for bin_index in range(n_bins):
            selected = bin_indices == bin_index
            count = int(np.count_nonzero(selected))
            if count:
                mean_confidence: float | None = float(np.mean(confidence[selected]))
                observed_frequency: float | None = float(np.mean(observed[selected]))
                gap: float | None = abs(mean_confidence - observed_frequency)
                weighted_error += (count / len(labels)) * gap
                gaps.append(gap)
            else:
                mean_confidence = None
                observed_frequency = None
                gap = None
            rows.append(
                {
                    "algorithm": algorithm,
                    "fold": fold,
                    "class": class_name,
                    "bin": bin_index,
                    "bin_lower": bin_index / n_bins,
                    "bin_upper": (bin_index + 1) / n_bins,
                    "sample_count": count,
                    "mean_confidence": mean_confidence,
                    "observed_frequency": observed_frequency,
                    "absolute_calibration_gap": gap,
                }
            )
        class_ece.append(float(weighted_error))
        class_mce.append(max(gaps, default=0.0))
    metrics = {
        "multiclass_expected_calibration_error": float(np.mean(class_ece)),
        "multiclass_maximum_calibration_error": float(max(class_mce)),
    }
    for class_name, ece, mce in zip(
        EXPECTED_CLASSES,
        class_ece,
        class_mce,
        strict=True,
    ):
        metrics[f"{class_name.lower()}_expected_calibration_error"] = ece
        metrics[f"{class_name.lower()}_maximum_calibration_error"] = mce
    return rows, metrics


def _verify_published_bins(
    published: object,
    expected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _require(isinstance(published, list), "D06 reliability_bins must be an array")
    _require(
        len(published) == len(expected_rows),
        "D06 reliability_bins count does not match the expected algorithm/fold/class/bin grid",
    )
    observed_by_key: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for item in published:
        _require(isinstance(item, dict), "D06 reliability bin must be an object")
        key = (
            str(item.get("algorithm", "")),
            _as_integer(item.get("fold"), "reliability bin fold"),
            str(item.get("class", "")),
            _as_integer(item.get("bin"), "reliability bin index"),
        )
        _require(
            key not in observed_by_key,
            f"duplicate D06 reliability bin key: {key}",
        )
        observed_by_key[key] = item
    expected_by_key = {
        (
            str(row["algorithm"]),
            int(row["fold"]),
            str(row["class"]),
            int(row["bin"]),
        ): row
        for row in expected_rows
    }
    _require(
        set(observed_by_key) == set(expected_by_key),
        "D06 reliability bin keys are missing, unexpected, or mislabelled",
    )

    audit_rows: list[dict[str, Any]] = []
    for key, expected in expected_by_key.items():
        observed = observed_by_key[key]
        prefix = f"reliability bin {key}"
        for name in ("bin_lower", "bin_upper"):
            _close(observed.get(name), float(expected[name]), f"{prefix} {name}")
        count = _as_integer(observed.get("sample_count"), f"{prefix} sample_count")
        _require(
            count == int(expected["sample_count"]),
            f"{prefix} sample_count mismatch",
        )
        row: dict[str, Any] = {
            "algorithm": key[0],
            "fold": key[1],
            "class": key[2],
            "bin": key[3],
            "bin_lower": expected["bin_lower"],
            "bin_upper": expected["bin_upper"],
            "published_sample_count": count,
            "recomputed_sample_count": expected["sample_count"],
        }
        for name in (
            "mean_confidence",
            "observed_frequency",
            "absolute_calibration_gap",
        ):
            observed_value = observed.get(name)
            expected_value = expected[name]
            if expected_value is None:
                _require(
                    observed_value is None,
                    f"{prefix} {name} must be null for an empty bin",
                )
                row[f"published_{name}"] = None
                row[f"recomputed_{name}"] = None
            else:
                _close(observed_value, float(expected_value), f"{prefix} {name}")
                row[f"published_{name}"] = float(observed_value)
                row[f"recomputed_{name}"] = float(expected_value)
        row["passed"] = True
        audit_rows.append(row)
    return audit_rows


def _verify_aggregate_metrics(
    aggregates: pd.DataFrame,
    metrics_by_algorithm: dict[str, list[dict[str, float]]],
) -> int:
    required_columns = {
        "task",
        "algorithm",
        "metric",
        "successful_folds",
        "mean",
        "std",
        "min",
        "max",
    }
    _require(
        required_columns.issubset(aggregates.columns),
        "D06 aggregate metric schema is incomplete",
    )
    _require(
        not aggregates.duplicated(["algorithm", "metric"]).any(),
        "D06 aggregate metrics contain duplicate algorithm/metric rows",
    )
    verified = 0
    for algorithm, fold_records in metrics_by_algorithm.items():
        _require(bool(fold_records), f"{algorithm} has no independently audited folds")
        for metric in _CALIBRATION_METRICS:
            _require(
                all(metric in record for record in fold_records),
                f"{algorithm} independent audit omitted aggregate metric {metric}",
            )
            selected = aggregates.loc[
                (aggregates["algorithm"].astype(str) == algorithm)
                & (aggregates["metric"].astype(str) == metric)
            ]
            _require(
                len(selected) == 1,
                f"D06 aggregate metric is missing or duplicated: {algorithm}/{metric}",
            )
            row = selected.iloc[0]
            _require(
                str(row["task"]) == "classification",
                "D06 calibration aggregate task must be classification",
            )
            values = np.asarray(
                [record[metric] for record in fold_records],
                dtype=np.float64,
            )
            _require(
                _as_integer(row["successful_folds"], "aggregate successful_folds")
                == len(values),
                f"aggregate fold count mismatch for {algorithm}/{metric}",
            )
            expected = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, value in expected.items():
                _close(row[name], value, f"aggregate {algorithm}/{metric}/{name}")
            verified += 1
    return verified


def audit_calibration_reliability(
    policy_run_dir: Path,
    dataset_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_policy_calibration_audits"),
) -> dict[str, Any]:
    """Re-fit D06.3 temperature, reproduce its threshold and rebuild OOS reliability."""
    policy_run_dir = policy_run_dir.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    _require(policy_run_dir.is_dir(), f"D06 policy run directory not found: {policy_run_dir}")
    _require(dataset_path.is_file(), f"D04 dataset not found: {dataset_path}")

    d06_manifest, verified_sources = verify_manifest_outputs(
        policy_run_dir,
        "decision_policy_manifest.json",
        _D06_OUTPUTS,
    )
    _require(
        d06_manifest.get("contract_version") == "D06.3",
        "D07.3 requires the D06.3 contract",
    )
    d06_summary = read_json_object(policy_run_dir / "decision_policy_summary.json")
    dataset, dates, _ = load_execution_dataset(dataset_path)
    d04_manifest_path = dataset_path.parent / "execution_dataset_manifest.json"
    _require(d04_manifest_path.is_file(), "D04 execution dataset manifest is missing")
    d04_hash = sha256_file(dataset_path)
    d04_manifest_hash = sha256_file(d04_manifest_path)
    d06_input = d06_manifest.get("input")
    _require(isinstance(d06_input, dict), "D06 input manifest is missing")
    _require(
        d06_input.get("execution_dataset") == dataset_path.name
        and d06_input.get("execution_dataset_sha256") == d04_hash,
        "D06 and D04 execution dataset identities differ",
    )
    _require(
        d06_input.get("source_manifest_sha256") == d04_manifest_hash,
        "D06 and D04 source manifest identities differ",
    )
    _require(
        d06_summary.get("snapshot_file") == dataset_path.name
        and d06_summary.get("snapshot_sha256") == d04_hash,
        "D06 summary snapshot identity differs from D04",
    )

    configuration = d06_manifest.get("configuration")
    _require(isinstance(configuration, dict), "D06 configuration is missing")
    algorithms = d06_summary.get("algorithms")
    _require(
        isinstance(algorithms, list)
        and bool(algorithms)
        and all(isinstance(value, str) and value for value in algorithms)
        and len(set(algorithms)) == len(algorithms),
        "D06 algorithm list is invalid",
    )
    fold_count = _as_integer(d06_summary.get("fold_count"), "D06 fold_count")
    _require(fold_count > 0, "D06 fold_count must be positive")
    _require(
        d06_summary.get("status") == "COMPLETED"
        and d06_summary.get("failed_model_folds") == 0
        and d06_summary.get("completed_model_folds") == len(algorithms) * fold_count,
        "D06 run is not fully completed",
    )
    test_fraction = _as_finite(
        configuration.get("test_fraction"),
        "manifest test_fraction",
    )
    _require(
        0.0 < test_fraction < 0.5,
        "D06 test_fraction is outside its supported range",
    )
    expected_folds = build_expanding_folds(
        len(dataset.features),
        n_folds=fold_count,
        test_fraction=test_fraction,
    )
    _require(
        configuration.get("algorithms") == algorithms
        and _as_integer(configuration.get("fold_count"), "manifest fold_count") == fold_count,
        "D06 manifest and summary algorithm/fold declarations differ",
    )
    _require(
        d06_summary.get("classes") == list(EXPECTED_CLASSES)
        and configuration.get("classes") == list(EXPECTED_CLASSES),
        "D06 class ordering differs from the canonical class contract",
    )
    _require(
        d06_summary.get("probability_calibration_method")
        == "multiclass_temperature_scaling"
        and configuration.get("probability_calibration_method")
        == "multiclass_temperature_scaling",
        "D06 calibration method is not recognized",
    )
    bounds_raw = configuration.get("probability_calibration_temperature_bounds")
    _require(
        isinstance(bounds_raw, list) and len(bounds_raw) == 2,
        "D06 temperature bounds are missing",
    )
    bounds = (
        _as_finite(bounds_raw[0], "minimum temperature"),
        _as_finite(bounds_raw[1], "maximum temperature"),
    )
    _require(0.0 < bounds[0] < bounds[1], "D06 temperature bounds are invalid")
    summary_bounds = d06_summary.get("probability_calibration_temperature_bounds")
    _require(
        isinstance(summary_bounds, list)
        and len(summary_bounds) == 2
        and np.allclose(np.asarray(summary_bounds, dtype=float), bounds, rtol=0.0, atol=1e-12),
        "D06 manifest and summary temperature bounds differ",
    )
    n_bins = _as_integer(
        configuration.get("reliability_bin_count"),
        "reliability_bin_count",
    )
    summary_bin_count = _as_integer(
        d06_summary.get("reliability_bin_count"),
        "summary reliability_bin_count",
    )
    _require(
        n_bins >= 2 and summary_bin_count == n_bins,
        "D06 reliability bin count is invalid or inconsistent",
    )
    grid_raw = configuration.get("threshold_grid")
    _require(isinstance(grid_raw, list) and bool(grid_raw), "D06 threshold grid is missing")
    threshold_grid = tuple(_as_finite(value, "threshold grid value") for value in grid_raw)
    _require(
        all(0.0 <= value <= 1.0 for value in threshold_grid)
        and len(set(threshold_grid)) == len(threshold_grid),
        "D06 threshold grid contains invalid or duplicate values",
    )
    summary_grid = d06_summary.get("threshold_grid")
    _require(
        isinstance(summary_grid, list)
        and np.allclose(
            np.asarray(summary_grid, dtype=float),
            np.asarray(threshold_grid),
            rtol=0.0,
            atol=1e-12,
        ),
        "D06 manifest and summary threshold grids differ",
    )
    _require(
        configuration.get("selection_objective")
        == d06_summary.get("threshold_selection_objective")
        == "macro_f1",
        "D06 threshold-selection objective is not recognized",
    )
    calibration_fraction = _as_finite(
        d06_summary.get("calibration_fraction"),
        "calibration_fraction",
    )
    _require(
        0.05 < calibration_fraction < 0.5
        and np.isclose(
            _as_finite(configuration.get("calibration_fraction"), "manifest calibration_fraction"),
            calibration_fraction,
            rtol=0.0,
            atol=1e-12,
        ),
        "D06 calibration fraction is invalid or inconsistent",
    )
    output_contract = d06_summary.get("outputs")
    _require(
        isinstance(output_contract, dict)
        and output_contract.get("calibration_predictions") == "calibration_predictions.csv"
        and output_contract.get("predictions") == "oos_predictions.csv",
        "D06 summary does not declare the D07.3 raw probability artifacts",
    )

    folds = pd.read_csv(policy_run_dir / "fold_metrics.csv", float_precision="round_trip")
    aggregates = pd.read_csv(policy_run_dir / "aggregate_metrics.csv", float_precision="round_trip")
    oos = pd.read_csv(policy_run_dir / "oos_predictions.csv", float_precision="round_trip")
    calibration = pd.read_csv(
        policy_run_dir / "calibration_predictions.csv",
        float_precision="round_trip",
    )
    _require(
        len(folds) == len(algorithms) * fold_count
        and len(folds) > 0
        and not folds.duplicated(["algorithm", "fold"]).any(),
        "D06 fold metrics have missing or duplicate algorithm/fold rows",
    )
    _require(
        len(oos) == _as_integer(d06_summary.get("prediction_rows"), "D06 prediction_rows")
        and not oos.duplicated(["algorithm", "fold", "row_index"]).any(),
        "D06 OOS predictions have missing or duplicate rows",
    )
    _require(
        not calibration.duplicated(["algorithm", "fold", "row_index"]).any(),
        "D06 calibration probabilities contain duplicate row keys",
    )
    _require(
        set(folds["algorithm"].astype(str)) == set(algorithms)
        and set(oos["algorithm"].astype(str)) == set(algorithms)
        and set(calibration["algorithm"].astype(str)) == set(algorithms),
        "D06 algorithms differ across fold and probability artifacts",
    )
    _require(
        {
            "fold",
            "algorithm",
            "row_index",
            "feature_date",
            "actual",
            *_RAW_COLUMNS,
        }.issubset(calibration.columns),
        "D06 calibration prediction schema is incomplete",
    )
    _require(
        {
            "fold",
            "algorithm",
            "row_index",
            "feature_date",
            "actual",
            "argmax_prediction",
            "policy_prediction",
            "selected_neutral_threshold",
            *_RAW_COLUMNS,
            *_CALIBRATED_COLUMNS,
        }.issubset(oos.columns),
        "D06 OOS prediction schema is incomplete for D07.3",
    )
    _require(
        {
            "fold",
            "algorithm",
            "status",
            "task",
            "train_start_index",
            "train_end_index",
            "test_start_index",
            "test_end_index",
            "calibration_start_index",
            "calibration_end_index",
            "calibration_samples",
            "test_samples",
            "probability_calibration_temperature",
            "selected_neutral_threshold",
        }.issubset(folds.columns),
        "D06 fold metrics are missing D07.3 calibration fields",
    )
    folds = folds.copy()
    oos = oos.copy()
    calibration = calibration.copy()
    for frame, columns, label in (
        (folds, ("fold", "train_start_index", "train_end_index", "test_start_index",
                 "test_end_index", "calibration_start_index", "calibration_end_index",
                 "calibration_samples", "test_samples"), "fold metrics"),
        (oos, ("fold", "row_index"), "OOS predictions"),
        (calibration, ("fold", "row_index"), "calibration predictions"),
    ):
        for column in columns:
            frame[column] = frame[column].map(
                lambda value, name=f"{label} {column}": _as_integer(value, name)
            )

    fold_audit_rows: list[dict[str, Any]] = []
    expected_bin_rows: list[dict[str, Any]] = []
    metrics_by_algorithm: dict[str, list[dict[str, float]]] = {
        str(algorithm): [] for algorithm in algorithms
    }
    expected_total_oos = 0
    expected_total_calibration = 0
    for algorithm in algorithms:
        for fold_id in range(1, fold_count + 1):
            fold_selected = folds.loc[
                (folds["algorithm"].astype(str) == algorithm)
                & (folds["fold"] == fold_id)
            ]
            _require(len(fold_selected) == 1, f"{algorithm} fold {fold_id} metrics row is missing")
            fold_row = fold_selected.iloc[0]
            _require(
                str(fold_row["status"]) == "COMPLETED"
                and str(fold_row["task"]) == "classification",
                f"{algorithm} fold {fold_id} is not a completed classifier evaluation",
            )
            train_start = int(fold_row["train_start_index"])
            train_end = int(fold_row["train_end_index"])
            test_start = int(fold_row["test_start_index"])
            test_end = int(fold_row["test_end_index"])
            expected_fold = expected_folds[fold_id - 1]
            _require(
                (train_start, train_end, test_start, test_end)
                == (
                    expected_fold.train_start,
                    expected_fold.train_end,
                    expected_fold.test_start,
                    expected_fold.test_end,
                ),
                f"{algorithm} fold {fold_id} differs from the declared expanding-fold geometry",
            )
            calibration_start = int(fold_row["calibration_start_index"])
            calibration_end = int(fold_row["calibration_end_index"])
            calibration_count = int(fold_row["calibration_samples"])
            _require(
                train_start == 0
                and train_end == test_start
                and 0 <= calibration_start < calibration_end == train_end
                and calibration_count == calibration_end - calibration_start
                and test_start < test_end <= len(dataset.target),
                f"{algorithm} fold {fold_id} temporal boundaries are invalid",
            )
            expected_calibration_count = max(2, int(math.ceil(train_end * calibration_fraction)))
            _require(
                calibration_count == expected_calibration_count,
                f"{algorithm} fold {fold_id} calibration suffix differs from D06 configuration",
            )
            expected_calibration_indices = np.arange(
                calibration_start,
                calibration_end,
                dtype=np.int64,
            )
            cal_rows = calibration.loc[
                (calibration["algorithm"].astype(str) == algorithm)
                & (calibration["fold"] == fold_id)
            ].sort_values("row_index")
            _require(
                np.array_equal(
                    cal_rows["row_index"].to_numpy(dtype=np.int64),
                    expected_calibration_indices,
                ),
                f"{algorithm} fold {fold_id} calibration row coverage is incomplete",
            )
            calibration_actual = np.asarray(dataset.target, dtype=str)[
                expected_calibration_indices
            ]
            _require(
                np.array_equal(cal_rows["actual"].astype(str).to_numpy(), calibration_actual),
                f"{algorithm} fold {fold_id} calibration labels differ from D04",
            )
            expected_calibration_dates = np.asarray(dates, dtype=str)[
                expected_calibration_indices
            ]
            _require(
                np.array_equal(cal_rows["feature_date"].astype(str).to_numpy(), expected_calibration_dates),
                f"{algorithm} fold {fold_id} calibration feature dates differ from D04",
            )
            raw_calibration = _probability_matrix(
                cal_rows,
                _RAW_COLUMNS,
                f"{algorithm} fold {fold_id} calibration",
            )
            temperature, calibration_scores = _fit_temperature_independently(
                calibration_actual,
                raw_calibration,
                bounds,
            )
            recorded_temperature = _as_finite(
                fold_row["probability_calibration_temperature"],
                "probability_calibration_temperature",
            )
            _require(
                bounds[0] <= recorded_temperature <= bounds[1],
                f"{algorithm} fold {fold_id} temperature is outside declared bounds",
            )
            _close(
                recorded_temperature,
                temperature,
                f"{algorithm} fold {fold_id} independently fitted temperature",
            )
            scaled_calibration = _apply_temperature(raw_calibration, temperature)
            selected = _select_threshold_independently(
                calibration_actual,
                scaled_calibration,
                threshold_grid,
            )
            for metric, value in calibration_scores.items():
                _close(
                    fold_row.get(metric),
                    value,
                    f"{algorithm} fold {fold_id} {metric}",
                )
            for metric, value in selected.items():
                _close(
                    fold_row.get(metric),
                    value,
                    f"{algorithm} fold {fold_id} {metric}",
                )

            expected_oos_indices = np.arange(test_start, test_end, dtype=np.int64)
            oos_rows = oos.loc[
                (oos["algorithm"].astype(str) == algorithm)
                & (oos["fold"] == fold_id)
            ].sort_values("row_index")
            _require(
                np.array_equal(
                    oos_rows["row_index"].to_numpy(dtype=np.int64),
                    expected_oos_indices,
                ),
                f"{algorithm} fold {fold_id} OOS row coverage is incomplete",
            )
            oos_actual = np.asarray(dataset.target, dtype=str)[expected_oos_indices]
            _require(
                np.array_equal(oos_rows["actual"].astype(str).to_numpy(), oos_actual),
                f"{algorithm} fold {fold_id} OOS labels differ from D04",
            )
            expected_oos_dates = np.asarray(dates, dtype=str)[expected_oos_indices]
            _require(
                np.array_equal(oos_rows["feature_date"].astype(str).to_numpy(), expected_oos_dates),
                f"{algorithm} fold {fold_id} OOS feature dates differ from D04",
            )
            raw_oos = _probability_matrix(
                oos_rows,
                _RAW_COLUMNS,
                f"{algorithm} fold {fold_id} raw OOS",
            )
            recomputed_oos = _apply_temperature(raw_oos, temperature)
            published_oos = _probability_matrix(
                oos_rows,
                _CALIBRATED_COLUMNS,
                f"{algorithm} fold {fold_id} calibrated OOS",
            )
            _require(
                np.allclose(recomputed_oos, published_oos, rtol=1e-7, atol=1e-9),
                (
                    f"{algorithm} fold {fold_id} OOS probabilities do not match "
                    "the independently applied temperature"
                ),
            )
            argmax = np.asarray(EXPECTED_CLASSES, dtype=str)[
                np.argmax(recomputed_oos, axis=1)
            ]
            policy = _neutral_decisions(
                recomputed_oos,
                selected["selected_neutral_threshold"],
            )
            _require(
                np.array_equal(oos_rows["argmax_prediction"].astype(str).to_numpy(), argmax),
                (
                    f"{algorithm} fold {fold_id} argmax predictions do not match "
                    "calibrated probabilities"
                ),
            )
            _require(
                np.array_equal(oos_rows["policy_prediction"].astype(str).to_numpy(), policy),
                (
                    f"{algorithm} fold {fold_id} neutral-policy predictions do not "
                    "match the independently selected threshold"
                ),
            )
            _require(
                np.allclose(
                    pd.to_numeric(oos_rows["selected_neutral_threshold"]).to_numpy(dtype=float),
                    selected["selected_neutral_threshold"],
                    rtol=0.0,
                    atol=1e-12,
                ),
                f"{algorithm} fold {fold_id} OOS threshold metadata is inconsistent",
            )

            oos_log_loss = _log_loss(oos_actual, recomputed_oos)
            oos_brier = _brier_score(oos_actual, recomputed_oos)
            bin_rows, reliability_scores = _reliability_diagnostics(
                oos_actual,
                recomputed_oos,
                algorithm=algorithm,
                fold=fold_id,
                n_bins=n_bins,
            )
            expected_bin_rows.extend(bin_rows)
            recomputed_metrics: dict[str, float] = {
                "probability_calibration_temperature": temperature,
                **selected,
                **calibration_scores,
                "multiclass_log_loss": oos_log_loss,
                "multiclass_brier_score": oos_brier,
                **reliability_scores,
            }
            for metric, value in recomputed_metrics.items():
                _close(
                    fold_row.get(metric),
                    value,
                    f"{algorithm} fold {fold_id} {metric}",
                )
            metrics_by_algorithm[algorithm].append(recomputed_metrics)
            fold_audit_rows.append(
                {
                    "algorithm": algorithm,
                    "fold": fold_id,
                    "calibration_start_index": calibration_start,
                    "calibration_end_index": calibration_end,
                    "calibration_samples": calibration_count,
                    "test_start_index": test_start,
                    "test_end_index": test_end,
                    "test_samples": len(oos_rows),
                    "published_temperature": recorded_temperature,
                    "recomputed_temperature": temperature,
                    "published_threshold": _as_finite(
                        fold_row["selected_neutral_threshold"],
                        "selected_neutral_threshold",
                    ),
                    "recomputed_threshold": selected["selected_neutral_threshold"],
                    **recomputed_metrics,
                }
            )
            expected_total_calibration += len(cal_rows)
            expected_total_oos += len(oos_rows)

    expected_bins = len(algorithms) * fold_count * len(EXPECTED_CLASSES) * n_bins
    _require(
        len(expected_bin_rows) == expected_bins,
        "independent reliability reconstruction produced an unexpected bin count",
    )
    bin_audit_rows = _verify_published_bins(
        d06_summary.get("reliability_bins"),
        expected_bin_rows,
    )
    aggregate_count = _verify_aggregate_metrics(aggregates, metrics_by_algorithm)

    checks = [
        {
            "name": "d06_manifest_and_d04_identity",
            "passed": True,
            "detail": "D06.3 source hashes/byte counts and D04 dataset/manifest identity verified",
        },
        {
            "name": "calibration_suffix_alignment",
            "passed": True,
            "detail": (
                "Every raw calibration probability, feature date and actual label "
                "maps to the D04 suffix"
            ),
        },
        {
            "name": "temperature_refit_and_calibration_scores",
            "passed": True,
            "detail": (
                "Temperature and raw/scaled calibration log-loss and Brier scores "
                "independently reproduced"
            ),
        },
        {
            "name": "threshold_selection_reproduction",
            "passed": True,
            "detail": (
                "Macro-F1 threshold selection and deterministic tie breaks "
                "independently reproduced"
            ),
        },
        {
            "name": "oos_temperature_application",
            "passed": True,
            "detail": (
                "Every raw OOS probability vector independently temperature-scaled "
                "to the published calibrated vector"
            ),
        },
        {
            "name": "oos_metrics_and_reliability",
            "passed": True,
            "detail": (
                "OOS log-loss, Brier score, all one-vs-rest bins, ECE and MCE "
                "independently recomputed"
            ),
        },
        {
            "name": "aggregate_calibration_metrics",
            "passed": True,
            "detail": (
                f"{aggregate_count} calibration/decision metric aggregate rows "
                "independently recomputed"
            ),
        },
    ]
    ticker = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(d06_summary.get("ticker", "UNKNOWN")),
    ).lower()
    run_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(d06_summary.get("run_id", policy_run_dir.name)),
    ).lower()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = output_root / f"{ticker}_{run_id}_calibration_audit_{timestamp}"
    audit_dir.mkdir(parents=False, exist_ok=False)

    fold_audit_path = audit_dir / "calibration_fold_audit.csv"
    bins_audit_path = audit_dir / "reliability_bins_audit.csv"
    summary_path = audit_dir / "calibration_audit_summary.json"
    atomic_write_csv(fold_audit_path, pd.DataFrame(fold_audit_rows))
    atomic_write_csv(bins_audit_path, pd.DataFrame(bin_audit_rows))
    audit_summary = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "audit_type": "D07.3_D06_calibration_and_reliability",
        "created_at": datetime.now(UTC).isoformat(),
        "ticker": str(d06_summary.get("ticker", "UNKNOWN")),
        "decision_policy_run_id": policy_run_dir.name,
        "execution_dataset": dataset_path.name,
        "execution_dataset_sha256": d04_hash,
        "source_manifest_sha256": d04_manifest_hash,
        "d06_manifest_sha256": sha256_file(policy_run_dir / "decision_policy_manifest.json"),
        "algorithm_count": len(algorithms),
        "algorithms": algorithms,
        "fold_count": fold_count,
        "calibration_rows_audited": expected_total_calibration,
        "oos_rows_audited": expected_total_oos,
        "reliability_bins_audited": len(bin_audit_rows),
        "checks_total": len(checks),
        "checks_passed": len(checks),
        "checks": checks,
        "interpretation": {
            "ece": (
                "Classwise one-vs-rest ECE is weighted by bin count over all OOS rows; "
                "multiclass ECE is the mean of the three classwise ECE values."
            ),
            "mce": (
                "Classwise MCE is the largest non-empty-bin absolute calibration gap; "
                "multiclass MCE is the maximum across classes."
            ),
            "temperature": (
                "D07.3 re-fits the scalar temperature from saved raw calibration-suffix "
                "probabilities; it then independently applies that temperature to "
                "saved raw OOS probabilities."
            ),
        },
        "limitations": [
            (
                "Calibration and threshold selection use the same historical suffix; "
                "the suffix scores are descriptive and optimistic."
            ),
            (
                "Equal-width reliability bins are sample-size dependent and can be "
                "noisy in sparse bins."
            ),
            (
                "A single global temperature cannot guarantee classwise calibration "
                "or protect against distribution shift."
            ),
            (
                "Historical OOS folds may have been inspected and are not a "
                "prospective untouched test."
            ),
            (
                "SHA-256 manifests detect file changes but are unsigned and do not "
                "authenticate artifact origin."
            ),
            (
                "Passing this audit verifies reproducibility and internal consistency, "
                "not predictive edge or trading profitability."
            ),
        ],
        "output_directory": str(audit_dir),
        "outputs": {
            "fold_audit": fold_audit_path.name,
            "reliability_bins_audit": bins_audit_path.name,
            "summary": summary_path.name,
            "manifest": "calibration_audit_manifest.json",
        },
    }
    atomic_write_json(summary_path, audit_summary)
    output_paths = (fold_audit_path, bins_audit_path, summary_path)
    audit_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "decision_policy_run_id": policy_run_dir.name,
            "d06_manifest_sha256": sha256_file(policy_run_dir / "decision_policy_manifest.json"),
            "d06_verified_sources": verified_sources,
            "execution_dataset": dataset_path.name,
            "execution_dataset_sha256": d04_hash,
            "source_manifest_sha256": d04_manifest_hash,
        },
        "outputs": manifest_output_entries(output_paths),
    }
    atomic_write_json(audit_dir / "calibration_audit_manifest.json", audit_manifest)
    return audit_summary
