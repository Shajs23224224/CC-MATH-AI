"""Audit D06 decision-policy predictions against the verified D04 execution dataset."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from scripts.run_execution_ml_walk_forward import (
    EXPECTED_CLASSES,
    load_execution_dataset,
)

_REQUIRED_D06_OUTPUTS = (
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "oos_predictions.csv",
    "decision_policy_summary.json",
)
_REQUIRED_DATASET_COLUMNS = (
    "entry_date",
    "entry_open",
    "exit_close",
    "target_execution_return",
    "target_execution_direction",
)
_AGGREGATE_METRICS = (
    "argmax_accuracy",
    "policy_accuracy",
    "argmax_balanced_accuracy",
    "policy_balanced_accuracy",
    "argmax_macro_f1",
    "policy_macro_f1",
    "policy_neutral_precision",
    "policy_neutral_recall",
)


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


def _record_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise ValueError(f"D07.1 audit failed [{name}]: {detail}")


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        raise ValueError(f"{field} contains a missing boolean value")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} contains an invalid boolean value: {value!r}")


def _classification_metrics(
    prefix: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Recompute D06 fixed-class metrics using the original metric contract."""
    y_true = np.asarray(actual, dtype=str)
    y_pred = np.asarray(predicted, dtype=str)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("actual and predicted labels must be non-empty and aligned")
    if not set(y_true.tolist()).issubset(EXPECTED_CLASSES):
        raise ValueError("actual labels contain an unknown class")
    if not set(y_pred.tolist()).issubset(EXPECTED_CLASSES):
        raise ValueError("predictions contain an unknown class")

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
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
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
        metrics[f"{prefix}_{label.lower()}_f1"] = float(f1[index])
        metrics[f"{prefix}_{label.lower()}_support"] = float(support[index])
        metrics[f"{prefix}_{label.lower()}_predicted_count"] = float(predicted_count)
        metrics[f"{prefix}_{label.lower()}_predicted_share"] = float(predicted_count / len(y_true))

    for actual_index, actual_label in enumerate(EXPECTED_CLASSES):
        for predicted_index, predicted_label in enumerate(EXPECTED_CLASSES):
            metrics[
                f"{prefix}_confusion_{actual_label.lower()}_predicted_{predicted_label.lower()}"
            ] = float(matrix[actual_index, predicted_index])

    return metrics


def _apply_neutral_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 3
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("probability matrix is invalid")
    if isinstance(threshold, bool) or not math.isfinite(float(threshold)):
        raise ValueError("neutral threshold must be finite")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("neutral threshold must be between zero and one")

    result = []
    for down_probability, neutral_probability, up_probability in matrix:
        if neutral_probability >= threshold:
            result.append("NEUTRAL")
        elif down_probability >= up_probability:
            result.append("DOWN")
        else:
            result.append("UP")
    return np.asarray(result, dtype=str)


def _check_metric_record(
    stored: pd.Series,
    computed: dict[str, float],
    *,
    algorithm: str,
    fold: int,
    prefix: str,
) -> None:
    for name, expected in computed.items():
        if name not in stored.index:
            raise ValueError(f"fold metrics missing {name} for {algorithm} fold {fold}")
        observed = float(stored[name])
        if not math.isclose(
            observed,
            expected,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"{prefix} metric mismatch for {algorithm} fold {fold}: "
                f"{name} stored={observed:.12g}, recomputed={expected:.12g}"
            )


def _validate_aggregate_metrics(
    fold_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    algorithms: tuple[str, ...],
    fold_count: int,
) -> None:
    required = {"algorithm", "metric", "successful_folds", "mean"}
    missing = required - set(aggregate_metrics.columns)
    if missing:
        raise ValueError(f"aggregate_metrics.csv is missing columns: {sorted(missing)}")

    for algorithm in algorithms:
        selected_folds = fold_metrics.loc[fold_metrics["algorithm"] == algorithm]
        if len(selected_folds) != fold_count:
            raise ValueError(f"unexpected fold count for {algorithm}")

        for metric_name in _AGGREGATE_METRICS:
            if metric_name not in selected_folds.columns:
                raise ValueError(f"fold metrics missing aggregate source {metric_name}")

            rows = aggregate_metrics.loc[
                (aggregate_metrics["algorithm"] == algorithm)
                & (aggregate_metrics["metric"] == metric_name)
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"aggregate metric {metric_name} is missing or duplicated for {algorithm}"
                )

            row = rows.iloc[0]
            expected_mean = float(selected_folds[metric_name].astype(float).mean())
            if int(row["successful_folds"]) != fold_count:
                raise ValueError(
                    f"aggregate successful_folds mismatch for {algorithm} {metric_name}"
                )
            if not math.isclose(
                float(row["mean"]),
                expected_mean,
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise ValueError(f"aggregate mean mismatch for {algorithm} {metric_name}")


def _build_algorithm_summary(
    algorithm: str,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    actual = predictions["actual"].to_numpy(dtype=str)
    argmax = predictions["argmax_prediction"].to_numpy(dtype=str)
    policy = predictions["policy_prediction"].to_numpy(dtype=str)

    argmax_metrics = _classification_metrics("argmax", actual, argmax)
    policy_metrics = _classification_metrics("policy", actual, policy)

    return {
        "algorithm": algorithm,
        "oos_rows": len(predictions),
        "actual_class_counts": {label: int(np.sum(actual == label)) for label in EXPECTED_CLASSES},
        "argmax_prediction_counts": {
            label: int(np.sum(argmax == label)) for label in EXPECTED_CLASSES
        },
        "policy_prediction_counts": {
            label: int(np.sum(policy == label)) for label in EXPECTED_CLASSES
        },
        "argmax_accuracy": argmax_metrics["argmax_accuracy"],
        "argmax_balanced_accuracy": argmax_metrics["argmax_balanced_accuracy"],
        "argmax_macro_f1": argmax_metrics["argmax_macro_f1"],
        "policy_accuracy": policy_metrics["policy_accuracy"],
        "policy_balanced_accuracy": policy_metrics["policy_balanced_accuracy"],
        "policy_macro_f1": policy_metrics["policy_macro_f1"],
        "policy_neutral_precision": policy_metrics["policy_neutral_precision"],
        "policy_neutral_recall": policy_metrics["policy_neutral_recall"],
        "mean_selected_neutral_threshold": float(predictions["selected_neutral_threshold"].mean()),
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


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


def audit_decision_policy_run(
    policy_run_dir: Path,
    dataset_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_policy_audits"),
) -> dict[str, Any]:
    """Verify D06 artifacts and enrich OOS predictions with D04 realized returns.

    This is an integrity/diagnostic audit. It does not simulate positions,
    transaction costs, capital allocation, or portfolio profitability.
    """
    policy_run_dir = policy_run_dir.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if not policy_run_dir.is_dir():
        raise FileNotFoundError(f"D06 run directory not found: {policy_run_dir}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"D04 execution dataset not found: {dataset_path}")

    checks: list[dict[str, Any]] = []
    policy_manifest_path = policy_run_dir / "decision_policy_manifest.json"
    if not policy_manifest_path.is_file():
        raise FileNotFoundError("D06 decision_policy_manifest.json not found")
    policy_manifest = _read_json(policy_manifest_path)

    output_entries = policy_manifest.get("outputs")
    if not isinstance(output_entries, dict):
        raise ValueError("D06 manifest outputs must be an object")

    for filename in _REQUIRED_D06_OUTPUTS:
        artifact_path = policy_run_dir / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"D06 output is missing: {filename}")
        output_entry = output_entries.get(filename)
        expected = output_entry.get("sha256") if isinstance(output_entry, dict) else None
        actual_hash = _sha256_file(artifact_path)
        _record_check(
            checks,
            f"sha256_{filename}",
            bool(expected) and actual_hash == expected,
            f"expected={expected}; actual={actual_hash}",
        )

        expected_bytes = output_entry.get("bytes") if isinstance(output_entry, dict) else None
        if expected_bytes is not None:
            _record_check(
                checks,
                f"bytes_{filename}",
                artifact_path.stat().st_size == int(expected_bytes),
                f"expected={expected_bytes}; actual={artifact_path.stat().st_size}",
            )

    d06_summary = _read_json(policy_run_dir / "decision_policy_summary.json")
    input_manifest = policy_manifest.get("input", {})
    source_manifest_path = dataset_path.parent / "execution_dataset_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("D04 execution_dataset_manifest.json not found")

    dataset_hash = _sha256_file(dataset_path)
    source_manifest_hash = _sha256_file(source_manifest_path)
    _record_check(
        checks,
        "input_dataset_sha256",
        input_manifest.get("execution_dataset") == dataset_path.name
        and input_manifest.get("execution_dataset_sha256") == dataset_hash
        and d06_summary.get("snapshot_file") == dataset_path.name
        and d06_summary.get("snapshot_sha256") == dataset_hash,
        f"dataset={dataset_path.name}; sha256={dataset_hash}",
    )
    _record_check(
        checks,
        "source_manifest_sha256",
        input_manifest.get("source_manifest_sha256") == source_manifest_hash,
        f"source_manifest_sha256={source_manifest_hash}",
    )
    _record_check(
        checks,
        "run_identity_and_status",
        d06_summary.get("run_id") == policy_run_dir.name
        and d06_summary.get("status") == "COMPLETED"
        and int(d06_summary.get("failed_model_folds", -1)) == 0
        and d06_summary.get("outer_test_used_for_selection") is False,
        f"run_id={d06_summary.get('run_id')}; status={d06_summary.get('status')}",
    )

    dataset, _, d04_manifest = load_execution_dataset(dataset_path)
    dataset_frame = pd.read_csv(
        dataset_path,
        index_col="feature_date",
        float_precision="round_trip",
    )
    dataset_frame.index = pd.to_datetime(dataset_frame.index, errors="coerce")
    if dataset_frame.index.isna().any():
        raise ValueError("D04 dataset contains invalid feature dates")

    missing_dataset_columns = set(_REQUIRED_DATASET_COLUMNS) - set(dataset_frame.columns)
    if missing_dataset_columns:
        raise ValueError(
            f"D04 dataset missing execution audit columns: "
            f"{sorted(missing_dataset_columns)}"
        )

    ticker = str(d04_manifest.get("experiment", {}).get("ticker", "UNKNOWN"))
    _record_check(
        checks,
        "ticker_consistency",
        str(d06_summary.get("ticker", "")).upper() == ticker.upper(),
        f"D04 ticker={ticker}; D06 ticker={d06_summary.get('ticker')}",
    )

    entry_open = pd.to_numeric(dataset_frame["entry_open"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    exit_close = pd.to_numeric(dataset_frame["exit_close"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    execution_return = pd.to_numeric(
        dataset_frame["target_execution_return"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(entry_open))
        or not np.all(np.isfinite(exit_close))
        or not np.all(np.isfinite(execution_return))
        or np.any(entry_open <= 0.0)
        or np.any(exit_close <= 0.0)
    ):
        raise ValueError("D04 execution prices/returns contain invalid values")

    calculated_return = exit_close / entry_open - 1.0
    _record_check(
        checks,
        "execution_return_formula",
        bool(
            np.allclose(
                execution_return,
                calculated_return,
                rtol=1e-10,
                atol=1e-12,
            )
        ),
        "target_execution_return equals exit_close / entry_open - 1",
    )

    threshold_fraction = float(
        d04_manifest.get("experiment", {}).get("threshold_fraction", float("nan"))
    )
    if not math.isfinite(threshold_fraction) or threshold_fraction < 0.0:
        raise ValueError("D04 manifest has an invalid threshold_fraction")
    expected_labels = np.where(
        execution_return > threshold_fraction,
        "UP",
        np.where(
            execution_return < -threshold_fraction,
            "DOWN",
            "NEUTRAL",
        ),
    )
    observed_dataset_labels = dataset_frame["target_execution_direction"].astype(str).to_numpy()
    _record_check(
        checks,
        "execution_target_label_rule",
        bool(np.array_equal(expected_labels, observed_dataset_labels)),
        f"threshold_fraction={threshold_fraction}",
    )

    algorithms_raw = d06_summary.get("algorithms")
    if (
        not isinstance(algorithms_raw, list)
        or not algorithms_raw
        or any(not isinstance(item, str) or not item for item in algorithms_raw)
        or len(set(algorithms_raw)) != len(algorithms_raw)
    ):
        raise ValueError("D06 summary has an invalid algorithm list")
    algorithms = tuple(algorithms_raw)
    fold_count = int(d06_summary.get("fold_count", 0))
    if fold_count < 1:
        raise ValueError("D06 summary has an invalid fold_count")

    threshold_grid = np.asarray(
        d06_summary.get("threshold_grid", []),
        dtype=np.float64,
    )
    if (
        threshold_grid.size == 0
        or not np.all(np.isfinite(threshold_grid))
        or np.any(threshold_grid < 0.0)
        or np.any(threshold_grid > 1.0)
    ):
        raise ValueError("D06 summary has an invalid threshold grid")

    manifest_grid = np.asarray(
        policy_manifest.get("configuration", {}).get("threshold_grid", []),
        dtype=np.float64,
    )
    _record_check(
        checks,
        "threshold_grid_consistency",
        manifest_grid.shape == threshold_grid.shape
        and bool(np.allclose(manifest_grid, threshold_grid, rtol=0.0, atol=1e-12)),
        "D06 summary and manifest use the same neutral-threshold grid",
    )

    if d06_summary.get("threshold_selection_objective") != "macro_f1":
        raise ValueError("D06 threshold selection objective is not macro_f1")

    fold_metrics = pd.read_csv(policy_run_dir / "fold_metrics.csv")
    aggregate_metrics = pd.read_csv(policy_run_dir / "aggregate_metrics.csv")
    predictions = pd.read_csv(
        policy_run_dir / "oos_predictions.csv",
        float_precision="round_trip",
    )

    required_fold_columns = {
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
        "calibration_start_index",
        "calibration_end_index",
        "calibration_samples",
        "calibration_test_used_for_selection",
        "selected_neutral_threshold",
    }
    missing_fold_columns = required_fold_columns - set(fold_metrics.columns)
    if missing_fold_columns:
        raise ValueError(f"D06 fold_metrics.csv missing columns: {sorted(missing_fold_columns)}")

    required_prediction_columns = {
        "fold",
        "algorithm",
        "row_index",
        "feature_date",
        "actual",
        "argmax_prediction",
        "policy_prediction",
        "selected_neutral_threshold",
        "probability_down",
        "probability_neutral",
        "probability_up",
    }
    missing_prediction_columns = required_prediction_columns - set(predictions.columns)
    if missing_prediction_columns:
        raise ValueError(
            f"D06 oos_predictions.csv missing columns: {sorted(missing_prediction_columns)}"
        )

    expected_model_folds = len(algorithms) * fold_count
    _record_check(
        checks,
        "completed_model_fold_count",
        len(fold_metrics) == expected_model_folds
        and set(fold_metrics["status"].astype(str)) == {"COMPLETED"}
        and int(d06_summary.get("attempted_model_folds", -1)) == expected_model_folds
        and int(d06_summary.get("completed_model_folds", -1)) == expected_model_folds,
        f"expected={expected_model_folds}; observed={len(fold_metrics)}",
    )
    _record_check(
        checks,
        "prediction_row_count",
        len(predictions) == int(d06_summary.get("prediction_rows", -1)),
        f"summary={d06_summary.get('prediction_rows')}; observed={len(predictions)}",
    )

    if set(fold_metrics["algorithm"].astype(str)) != set(algorithms):
        raise ValueError("D06 fold metrics algorithm set does not match summary")
    if set(predictions["algorithm"].astype(str)) != set(algorithms):
        raise ValueError("D06 predictions algorithm set does not match summary")

    for column in (
        "fold",
        "train_start_index",
        "train_end_index",
        "test_start_index",
        "test_end_index",
        "training_samples",
        "test_samples",
        "calibration_start_index",
        "calibration_end_index",
        "calibration_samples",
    ):
        fold_metrics[column] = pd.to_numeric(fold_metrics[column], errors="raise").astype(int)

    for column in ("fold", "row_index"):
        numeric = pd.to_numeric(predictions[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"D06 predictions contains invalid {column}")
        predictions[column] = numeric.astype(np.int64)

    if predictions.duplicated(["algorithm", "row_index"]).any():
        raise ValueError("D06 predictions contain duplicated algorithm/row_index keys")

    _record_check(
        checks,
        "unique_oos_prediction_keys",
        not predictions.duplicated(["algorithm", "row_index"]).any(),
        "no repeated row is evaluated twice for the same algorithm",
    )

    probability_columns = (
        "probability_down",
        "probability_neutral",
        "probability_up",
    )
    for column in probability_columns:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    probability_matrix = predictions.loc[:, probability_columns].to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(probability_matrix))
        or np.any(probability_matrix < 0.0)
        or np.any(probability_matrix > 1.0)
        or not np.allclose(
            probability_matrix.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise ValueError("D06 OOS probability matrix is invalid")
    _record_check(
        checks,
        "valid_probability_matrix",
        True,
        "all probabilities are finite, bounded, and each row sums to one",
    )

    if not set(predictions["actual"].astype(str)).issubset(EXPECTED_CLASSES):
        raise ValueError("D06 actual labels contain unknown classes")

    threshold_lookup: dict[tuple[str, int], float] = {}
    fold_audit_rows: list[dict[str, Any]] = []
    audited_frames: list[pd.DataFrame] = []
    for _, fold_row in fold_metrics.iterrows():
        algorithm = str(fold_row["algorithm"])
        fold_number = int(fold_row["fold"])
        key = (algorithm, fold_number)
        if key in threshold_lookup:
            raise ValueError(f"duplicated D06 fold metrics row: {key}")

        start = int(fold_row["test_start_index"])
        end = int(fold_row["test_end_index"])
        train_start = int(fold_row["train_start_index"])
        train_end = int(fold_row["train_end_index"])
        test_samples = int(fold_row["test_samples"])
        threshold = float(fold_row["selected_neutral_threshold"])

        if (
            train_start != 0
            or train_end != start
            or start < 1
            or end <= start
            or end > len(dataset.features)
            or test_samples != end - start
        ):
            raise ValueError(f"invalid outer temporal boundaries for {key}")

        calibration_start = int(fold_row["calibration_start_index"])
        calibration_end = int(fold_row["calibration_end_index"])
        calibration_samples = int(fold_row["calibration_samples"])
        if (
            calibration_start < train_start
            or calibration_end != train_end
            or calibration_start >= calibration_end
            or calibration_samples != calibration_end - calibration_start
        ):
            raise ValueError(f"invalid inner calibration boundaries for {key}")

        if _as_bool(
            fold_row["test_used_for_selection"],
            field="test_used_for_selection",
        ):
            raise ValueError(f"outer test was used for selection in {key}")
        if not _as_bool(
            fold_row["calibration_test_used_for_selection"],
            field="calibration_test_used_for_selection",
        ):
            raise ValueError(f"calibration suffix was not marked as selection data in {key}")

        if not np.any(np.isclose(threshold_grid, threshold, rtol=0.0, atol=1e-12)):
            raise ValueError(f"selected threshold is outside D06 grid for {key}")

        threshold_lookup[key] = threshold
        selected = predictions.loc[
            (predictions["algorithm"].astype(str) == algorithm)
            & (predictions["fold"] == fold_number)
        ].copy()
        if len(selected) != test_samples:
            raise ValueError(
                f"prediction count mismatch for {key}: expected {test_samples}, got {len(selected)}"
            )

        row_indices = selected["row_index"].to_numpy(dtype=np.int64)
        expected_indices = np.arange(start, end, dtype=np.int64)
        if not np.array_equal(np.sort(row_indices), expected_indices):
            raise ValueError(f"OOS row-index coverage mismatch for {key}")

        selected = selected.sort_values("row_index").reset_index(drop=True)
        row_indices = selected["row_index"].to_numpy(dtype=np.int64)

        expected_dates = pd.to_datetime(
            dataset_frame.index.take(row_indices),
            utc=True,
            errors="coerce",
        )
        observed_dates = pd.to_datetime(
            selected["feature_date"],
            utc=True,
            errors="coerce",
        )
        if expected_dates.isna().any() or observed_dates.isna().any():
            raise ValueError(f"invalid OOS feature date for {key}")
        if not np.array_equal(expected_dates.asi8, observed_dates.array.asi8):
            raise ValueError(f"OOS feature date/index mismatch for {key}")

        actual = (
            dataset_frame["target_execution_direction"].iloc[row_indices].astype(str).to_numpy()
        )
        observed_actual = selected["actual"].astype(str).to_numpy()
        if not np.array_equal(actual, observed_actual):
            raise ValueError(f"D06 actual labels do not match D04 target for {key}")
        if set(actual.tolist()) != set(EXPECTED_CLASSES):
            raise ValueError(f"OOS test fold does not contain all classes for {key}")

        probabilities = selected.loc[:, probability_columns].to_numpy(dtype=np.float64)
        stored_thresholds = pd.to_numeric(
            selected["selected_neutral_threshold"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.allclose(
            stored_thresholds,
            threshold,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"prediction thresholds do not match fold metrics for {key}")

        expected_argmax = np.asarray(EXPECTED_CLASSES, dtype=object)[
            np.argmax(probabilities, axis=1)
        ].astype(str)
        observed_argmax = selected["argmax_prediction"].astype(str).to_numpy()
        if not np.array_equal(expected_argmax, observed_argmax):
            raise ValueError(f"argmax predictions cannot be reproduced for {key}")

        expected_policy = _apply_neutral_threshold(probabilities, threshold)
        observed_policy = selected["policy_prediction"].astype(str).to_numpy()
        if not np.array_equal(expected_policy, observed_policy):
            raise ValueError(f"policy predictions cannot be reproduced for {key}")

        _check_metric_record(
            fold_row,
            _classification_metrics("argmax", actual, expected_argmax),
            algorithm=algorithm,
            fold=fold_number,
            prefix="argmax",
        )
        _check_metric_record(
            fold_row,
            _classification_metrics("policy", actual, expected_policy),
            algorithm=algorithm,
            fold=fold_number,
            prefix="policy",
        )

        subset_frame = dataset_frame.iloc[row_indices]
        enriched = selected.copy()
        enriched["actual_execution_return"] = execution_return[row_indices]
        enriched["entry_date"] = subset_frame["entry_date"].astype(str).to_numpy()
        enriched["entry_open"] = entry_open[row_indices]
        enriched["exit_close"] = exit_close[row_indices]
        enriched["argmax_correct"] = expected_argmax == actual
        enriched["policy_correct"] = expected_policy == actual

        argmax_probability = np.asarray(
            [
                probabilities[index, EXPECTED_CLASSES.index(label)]
                for index, label in enumerate(expected_argmax)
            ],
            dtype=np.float64,
        )
        policy_probability = np.asarray(
            [
                probabilities[index, EXPECTED_CLASSES.index(label)]
                for index, label in enumerate(expected_policy)
            ],
            dtype=np.float64,
        )
        enriched["argmax_selected_probability"] = argmax_probability
        enriched["policy_selected_probability"] = policy_probability
        enriched["maximum_probability"] = probabilities.max(axis=1)
        sorted_probabilities = np.sort(probabilities, axis=1)
        enriched["probability_margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        enriched["policy_selected_class_probability"] = [
            probabilities[index, EXPECTED_CLASSES.index(label)]
            for index, label in enumerate(expected_policy)
        ]
        fold_record = {
            "algorithm": algorithm,
            "fold": fold_number,
            "train_start_index": train_start,
            "train_end_index": train_end,
            "calibration_start_index": calibration_start,
            "calibration_end_index": calibration_end,
            "test_start_index": start,
            "test_end_index": end,
            "test_samples": test_samples,
            "selected_neutral_threshold": threshold,
            "first_feature_date": str(selected["feature_date"].iloc[0]),
            "last_feature_date": str(selected["feature_date"].iloc[-1]),
        }
        fold_record.update(_classification_metrics("argmax", actual, expected_argmax))
        fold_record.update(_classification_metrics("policy", actual, expected_policy))
        fold_audit_rows.append(fold_record)

        audited_frames.append(enriched)

    expected_folds = set(range(1, fold_count + 1))
    for algorithm in algorithms:
        algorithm_folds = {
            int(value)
            for value in fold_metrics.loc[fold_metrics["algorithm"] == algorithm, "fold"].tolist()
        }
        if algorithm_folds != expected_folds:
            raise ValueError(f"fold identifiers are incomplete for {algorithm}")

        ordered = fold_metrics.loc[fold_metrics["algorithm"] == algorithm].sort_values("fold")
        previous_end = -1
        for _, row in ordered.iterrows():
            start = int(row["test_start_index"])
            end = int(row["test_end_index"])
            if start < previous_end:
                raise ValueError(f"outer test windows overlap for {algorithm}")
            previous_end = end

    _validate_aggregate_metrics(
        fold_metrics,
        aggregate_metrics,
        algorithms,
        fold_count,
    )
    _record_check(
        checks,
        "recomputed_fold_metrics",
        True,
        "stored argmax and D06 policy metrics match row-level OOS predictions",
    )
    _record_check(
        checks,
        "aggregate_metrics_consistency",
        True,
        "selected aggregate means match the means recomputed from fold metrics",
    )
    _record_check(
        checks,
        "temporal_isolation",
        True,
        (
            "outer test windows are chronological, non-overlapping, and excluded "
            "from threshold selection"
        ),
    )

    audited_predictions = pd.concat(audited_frames, ignore_index=True)
    audited_predictions = audited_predictions.sort_values(
        ["algorithm", "fold", "row_index"]
    ).reset_index(drop=True)

    per_algorithm = [
        _build_algorithm_summary(
            algorithm,
            audited_predictions.loc[audited_predictions["algorithm"] == algorithm],
        )
        for algorithm in algorithms
    ]

    all_returns = audited_predictions["actual_execution_return"].to_numpy(dtype=np.float64)
    summary = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "audit_type": "D07.1_D06_prediction_integrity",
        "created_at": datetime.now(UTC).isoformat(),
        "ticker": ticker,
        "decision_policy_run_id": policy_run_dir.name,
        "execution_dataset": dataset_path.name,
        "execution_dataset_sha256": dataset_hash,
        "source_manifest_sha256": source_manifest_hash,
        "rows_audited": len(audited_predictions),
        "algorithm_count": len(algorithms),
        "algorithms": list(algorithms),
        "fold_count": fold_count,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "checks": checks,
        "actual_execution_return": {
            "type": "gross next-session open-to-close target; before trading costs",
            "mean": float(np.mean(all_returns)),
            "std": float(np.std(all_returns, ddof=1)) if len(all_returns) > 1 else 0.0,
            "minimum": float(np.min(all_returns)),
            "maximum": float(np.max(all_returns)),
        },
        "per_algorithm": per_algorithm,
        "limitations": [
            "This audit validates data and prediction integrity; it is not a backtest.",
            "No position sizing, capital, fees, spread, slippage, or market impact is applied.",
            (
                "The historical OOS folds were previously inspected and are not "
                "an untouched prospective test."
            ),
            (
                "Joined realized execution returns do not establish profitability "
                "or future predictive performance."
            ),
        ],
        "outputs": {
            "predictions": "audited_predictions.csv",
            "fold_metrics": "fold_audit.csv",
            "summary": "audit_summary.json",
            "manifest": "audit_manifest.json",
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = output_root / (f"{ticker.lower()}_{policy_run_dir.name}_audit_{timestamp}")
    audit_dir.mkdir(parents=False, exist_ok=False)

    predictions_path = audit_dir / "audited_predictions.csv"
    fold_audit_path = audit_dir / "fold_audit.csv"
    summary_path = audit_dir / "audit_summary.json"

    _write_csv(predictions_path, audited_predictions)
    _write_csv(fold_audit_path, pd.DataFrame(fold_audit_rows))
    summary["output_directory"] = str(audit_dir)
    _write_json(summary_path, summary)

    output_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "decision_policy_run_id": policy_run_dir.name,
            "decision_policy_manifest_sha256": _sha256_file(policy_manifest_path),
            "execution_dataset": dataset_path.name,
            "execution_dataset_sha256": dataset_hash,
            "source_manifest_sha256": source_manifest_hash,
        },
        "outputs": {
            path.name: {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (predictions_path, fold_audit_path, summary_path)
        },
    }
    _write_json(audit_dir / "audit_manifest.json", output_manifest)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D07.1 integrity audit for D06 OOS decision-policy predictions."
    )
    parser.add_argument("--policy-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_policy_audits"),
    )
    args = parser.parse_args()

    try:
        summary = audit_decision_policy_run(
            args.policy_run,
            args.dataset,
            output_root=args.output_root,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            f"D07.1 AUDIT FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    print("\n========== D07.1 AUDIT SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
