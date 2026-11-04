"""D07.1: independently audit D06 predictions against the verified D04 dataset."""

from __future__ import annotations

import re
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

from execution.research.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    manifest_output_entries,
    read_json_object,
    sha256_file,
    verify_manifest_outputs,
)
from scripts.run_execution_decision_policy import (
    apply_neutral_threshold,
)
from scripts.run_execution_ml_walk_forward import (
    EXPECTED_CLASSES,
    load_execution_dataset,
)
from scripts.run_ml_walk_forward import build_expanding_folds

_D06_OUTPUTS = (
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "oos_predictions.csv",
    "decision_policy_summary.json",
)
_D04_COLUMNS = (
    "entry_date",
    "entry_open",
    "exit_close",
    "target_execution_return",
    "target_execution_direction",
)
_METRIC_PREFIXES = ("argmax", "policy")
_METADATA_COLUMNS = {
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_bool(value: object, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        raise ValueError(f"{field} contains a missing boolean")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} contains an invalid boolean: {value!r}")


def _as_integer_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    _require(np.all(np.isfinite(values)), f"{column} contains non-finite values")
    _require(np.all(values == np.floor(values)), f"{column} must contain integers")
    return values.astype(np.int64)


def _close(actual: object, expected: float, field: str) -> None:
    try:
        observed = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    _require(np.isfinite(observed), f"{field} is non-finite")
    _require(
        bool(np.isclose(observed, expected, rtol=1e-9, atol=1e-11)),
        f"{field} mismatch: stored={observed}, recomputed={expected}",
    )


def _classification_metrics(
    prefix: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Recompute D06 fixed-class metrics from OOS labels."""
    _require(len(actual) == len(predicted) > 0, "metric labels are empty or misaligned")
    _require(set(actual.tolist()).issubset(EXPECTED_CLASSES), "unknown actual class")
    _require(set(predicted.tolist()).issubset(EXPECTED_CLASSES), "unknown predicted class")

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(EXPECTED_CLASSES),
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=list(EXPECTED_CLASSES))
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(actual, predicted)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        f"{prefix}_macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=list(EXPECTED_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        f"{prefix}_prediction_unique_classes": float(np.unique(predicted).size),
        f"{prefix}_prediction_collapse": float(np.unique(predicted).size == 1),
    }
    for index, label in enumerate(EXPECTED_CLASSES):
        name = label.lower()
        predicted_count = int(matrix[:, index].sum())
        metrics[f"{prefix}_{name}_precision"] = float(precision[index])
        metrics[f"{prefix}_{name}_recall"] = float(recall[index])
        metrics[f"{prefix}_{name}_f1"] = float(f1[index])
        metrics[f"{prefix}_{name}_support"] = float(support[index])
        metrics[f"{prefix}_{name}_predicted_count"] = float(predicted_count)
        metrics[f"{prefix}_{name}_predicted_share"] = float(predicted_count / len(actual))
    for actual_index, actual_label in enumerate(EXPECTED_CLASSES):
        for predicted_index, predicted_label in enumerate(EXPECTED_CLASSES):
            key = f"{prefix}_confusion_{actual_label.lower()}_predicted_{predicted_label.lower()}"
            metrics[key] = float(matrix[actual_index, predicted_index])
    return metrics


def _verify_d04_dataset(
    dataset_path: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any], str, str]:
    dataset_path = dataset_path.expanduser().resolve()
    _require(dataset_path.is_file(), f"D04 dataset not found: {dataset_path}")
    manifest_path = dataset_path.parent / "execution_dataset_manifest.json"
    _require(manifest_path.is_file(), "D04 execution_dataset_manifest.json not found")

    manifest = read_json_object(manifest_path)
    artifact = manifest.get("artifact")
    _require(isinstance(artifact, dict), "D04 artifact entry is missing")
    _require(artifact.get("file") == dataset_path.name, "D04 dataset filename mismatch")
    dataset_hash = sha256_file(dataset_path)
    _require(artifact.get("sha256") == dataset_hash, "D04 dataset SHA-256 mismatch")
    expected_bytes = artifact.get("bytes")
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and dataset_path.stat().st_size == expected_bytes,
        "D04 dataset byte count mismatch",
    )

    dataset, dates, loaded_manifest = load_execution_dataset(dataset_path)
    _require(loaded_manifest == manifest, "D04 loader returned a different manifest")

    frame = pd.read_csv(
        dataset_path,
        index_col="feature_date",
        float_precision="round_trip",
    )
    _require(len(frame) == len(dataset.features), "D04 row count mismatch")
    missing = set(_D04_COLUMNS) - set(frame.columns)
    _require(not missing, f"D04 dataset missing columns: {sorted(missing)}")

    date_index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    _require(not date_index.isna().any(), "D04 feature dates contain invalid values")
    _require(not date_index.has_duplicates, "D04 feature dates contain duplicates")
    _require(date_index.is_monotonic_increasing, "D04 feature dates are not chronological")

    execution = (
        frame[["entry_open", "exit_close", "target_execution_return"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    _require(np.all(np.isfinite(execution)), "D04 prices/returns contain non-finite values")
    _require(np.all(execution[:, :2] > 0.0), "D04 prices must be positive")
    calculated = execution[:, 1] / execution[:, 0] - 1.0
    _require(
        bool(np.allclose(execution[:, 2], calculated, rtol=1e-10, atol=1e-12)),
        "D04 target return does not equal exit_close / entry_open - 1",
    )

    entry_dates = pd.to_datetime(frame["entry_date"], errors="coerce", utc=True)
    _require(not entry_dates.isna().any(), "D04 entry dates contain invalid values")
    _require(
        bool(np.all(entry_dates.to_numpy() > date_index.to_numpy())),
        "D04 entry dates must follow feature dates",
    )
    _require(entry_dates.is_monotonic_increasing, "D04 entry dates are not chronological")
    _require(not entry_dates.duplicated().any(), "D04 entry dates contain duplicates")

    threshold = manifest.get("experiment", {}).get("threshold_fraction")
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and np.isfinite(float(threshold))
        and 0.0 <= float(threshold) <= 1.0,
        "D04 threshold_fraction is invalid",
    )
    returns = execution[:, 2]
    labels = np.where(
        returns > float(threshold),
        "UP",
        np.where(returns < -float(threshold), "DOWN", "NEUTRAL"),
    )
    observed_labels = frame["target_execution_direction"].astype(str).to_numpy()
    _require(
        bool(np.array_equal(labels.astype(str), observed_labels)),
        "D04 direction labels do not match the configured threshold",
    )
    _require(
        bool(np.array_equal(observed_labels, np.asarray(dataset.target, dtype=str))),
        "D04 loader labels do not match the raw dataset",
    )
    return frame, dates, manifest, dataset_hash, sha256_file(manifest_path)


def _verify_d06_summary(
    summary: dict[str, Any],
) -> tuple[tuple[str, ...], int, int]:
    _require(summary.get("status") == "COMPLETED", "D06 run is not fully completed")
    _require(summary.get("failed_model_folds") == 0, "D06 reports failed model folds")
    algorithms_raw = summary.get("algorithms")
    _require(
        isinstance(algorithms_raw, list)
        and bool(algorithms_raw)
        and all(isinstance(item, str) and item for item in algorithms_raw)
        and len(set(algorithms_raw)) == len(algorithms_raw),
        "D06 algorithm list is invalid",
    )
    algorithms = tuple(algorithms_raw)
    fold_count = summary.get("fold_count")
    prediction_rows = summary.get("prediction_rows")
    _require(
        isinstance(fold_count, int) and not isinstance(fold_count, bool) and fold_count > 0,
        "D06 fold_count is invalid",
    )
    _require(
        isinstance(prediction_rows, int)
        and not isinstance(prediction_rows, bool)
        and prediction_rows > 0,
        "D06 prediction_rows is invalid",
    )
    expected_runs = len(algorithms) * fold_count
    _require(
        summary.get("attempted_model_folds") == expected_runs,
        "D06 attempted fold count mismatch",
    )
    _require(
        summary.get("completed_model_folds") == expected_runs,
        "D06 completed fold count mismatch",
    )
    _require(
        summary.get("outer_test_used_for_selection") is False,
        "D06 summary marks outer test data as selection data",
    )
    _require(
        summary.get("classes") == list(EXPECTED_CLASSES),
        "D06 class order differs from the expected canonical order",
    )
    _require(
        summary.get("threshold_selection_objective") == "macro_f1",
        "D06 threshold selection objective is not recognized",
    )
    return algorithms, fold_count, prediction_rows


def _verify_aggregate_metrics(
    folds: pd.DataFrame,
    aggregates: pd.DataFrame,
    algorithms: tuple[str, ...],
) -> int:
    required = {"algorithm", "metric", "successful_folds", "mean", "std", "min", "max"}
    _require(required.issubset(aggregates.columns), "D06 aggregate metrics schema is incomplete")
    _require(
        not aggregates.duplicated(["algorithm", "metric"]).any(),
        "D06 aggregate metrics contain duplicate algorithm/metric keys",
    )
    expected_rows: set[tuple[str, str]] = set()
    verified = 0
    for algorithm in algorithms:
        group = folds.loc[folds["algorithm"].astype(str) == algorithm]
        _require(not group.empty, f"D06 fold metrics are missing {algorithm}")
        numeric_columns = [
            column
            for column in group.columns
            if column not in _METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(group[column])
            and group[column].notna().all()
        ]
        for metric in numeric_columns:
            expected_rows.add((algorithm, metric))
            selected = aggregates.loc[
                (aggregates["algorithm"].astype(str) == algorithm)
                & (aggregates["metric"].astype(str) == metric)
            ]
            _require(len(selected) == 1, f"D06 aggregate row missing: {algorithm}/{metric}")
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=np.float64)
            row = selected.iloc[0]
            _require(int(row["successful_folds"]) == len(values), "aggregate fold count mismatch")
            expected = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, value in expected.items():
                _close(row[name], value, f"aggregate {algorithm}/{metric}/{name}")
            verified += 1
    observed_rows = {
        (str(row.algorithm), str(row.metric)) for row in aggregates.itertuples(index=False)
    }
    _require(
        observed_rows == expected_rows,
        "D06 aggregate metric keys are incomplete or unexpected",
    )
    return verified


def audit_decision_policy_run(
    policy_run_dir: Path,
    dataset_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_policy_audits"),
) -> dict[str, Any]:
    """Verify a completed D06 run and join each OOS prediction to D04 outcomes."""
    policy_run_dir = policy_run_dir.expanduser().resolve()
    _require(policy_run_dir.is_dir(), f"D06 run directory not found: {policy_run_dir}")

    d06_manifest, _ = verify_manifest_outputs(
        policy_run_dir,
        "decision_policy_manifest.json",
        _D06_OUTPUTS,
    )
    d06_summary = read_json_object(policy_run_dir / "decision_policy_summary.json")
    algorithms, fold_count, expected_prediction_rows = _verify_d06_summary(d06_summary)

    frame, dates, d04_manifest, dataset_hash, source_manifest_hash = _verify_d04_dataset(
        dataset_path
    )
    d06_input = d06_manifest.get("input")
    _require(isinstance(d06_input, dict), "D06 input manifest is missing")
    _require(
        d06_input.get("execution_dataset") == Path(dataset_path).name,
        "D06 and D04 dataset filenames differ",
    )
    _require(
        d06_input.get("execution_dataset_sha256") == dataset_hash,
        "D06 and D04 dataset hashes differ",
    )
    _require(
        d06_input.get("source_manifest_sha256") == source_manifest_hash,
        "D06 and D04 source-manifest hashes differ",
    )
    _require(
        d06_summary.get("snapshot_sha256") == dataset_hash,
        "D06 summary snapshot hash differs from D04",
    )
    _require(
        d06_manifest.get("configuration", {}).get("algorithms") == list(algorithms),
        "D06 manifest algorithms differ from its summary",
    )
    configuration = d06_manifest.get("configuration")
    _require(isinstance(configuration, dict), "D06 manifest configuration is missing")
    _require(
        configuration.get("algorithms") == list(algorithms),
        "D06 manifest algorithms differ from its summary",
    )
    _require(
        configuration.get("fold_count") == fold_count,
        "D06 manifest fold count differs from its summary",
    )
    test_fraction = configuration.get("test_fraction")
    _require(
        isinstance(test_fraction, (int, float))
        and not isinstance(test_fraction, bool)
        and np.isfinite(float(test_fraction))
        and 0.0 < float(test_fraction) < 0.5,
        "D06 manifest test_fraction is invalid",
    )
    threshold_grid_raw = configuration.get("threshold_grid")
    _require(
        isinstance(threshold_grid_raw, list) and bool(threshold_grid_raw),
        "D06 manifest threshold_grid is missing",
    )
    threshold_grid = np.asarray(threshold_grid_raw, dtype=np.float64)
    _require(
        np.all(np.isfinite(threshold_grid))
        and np.all((threshold_grid >= 0.0) & (threshold_grid <= 1.0))
        and len(np.unique(threshold_grid)) == len(threshold_grid),
        "D06 manifest threshold_grid is invalid",
    )
    summary_threshold_grid = np.asarray(
        d06_summary.get("threshold_grid", []),
        dtype=np.float64,
    )
    _require(
        summary_threshold_grid.shape == threshold_grid.shape
        and np.allclose(
            summary_threshold_grid,
            threshold_grid,
            rtol=0.0,
            atol=1e-12,
        ),
        "D06 manifest and summary threshold grids differ",
    )
    _require(
        configuration.get("selection_objective")
        == d06_summary.get("threshold_selection_objective"),
        "D06 manifest and summary threshold objectives differ",
    )
    calibration_fraction = d06_summary.get("calibration_fraction")
    _require(
        isinstance(calibration_fraction, (int, float))
        and not isinstance(calibration_fraction, bool)
        and np.isfinite(float(calibration_fraction))
        and 0.05 < float(calibration_fraction) < 0.5,
        "D06 calibration_fraction is invalid",
    )
    _require(
        np.isclose(
            float(configuration.get("calibration_fraction", np.nan)),
            float(calibration_fraction),
            rtol=0.0,
            atol=1e-12,
        ),
        "D06 manifest and summary calibration_fraction differ",
    )
    expected_folds = build_expanding_folds(
        len(frame),
        n_folds=fold_count,
        test_fraction=float(test_fraction),
    )
    expected_fold_map = {fold.fold: fold for fold in expected_folds}

    folds = pd.read_csv(policy_run_dir / "fold_metrics.csv")
    aggregates = pd.read_csv(policy_run_dir / "aggregate_metrics.csv")
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
        "test_start_date",
        "test_last_feature_date",
        "calibration_start_index",
        "calibration_end_index",
        "calibration_samples",
        "calibration_test_used_for_selection",
        "selected_neutral_threshold",
    }
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
    _require(
        required_fold_columns.issubset(folds.columns),
        "D06 fold metrics schema is incomplete",
    )
    _require(
        required_prediction_columns.issubset(predictions.columns),
        "D06 OOS prediction schema is incomplete",
    )
    _require(len(folds) == len(algorithms) * fold_count, "D06 fold row count mismatch")
    _require(
        len(predictions) == expected_prediction_rows,
        "D06 prediction row count differs from its summary",
    )
    _require(
        set(folds["algorithm"].astype(str)) == set(algorithms),
        "D06 fold algorithms differ from summary",
    )
    _require(
        set(predictions["algorithm"].astype(str)) == set(algorithms),
        "D06 prediction algorithms differ from summary",
    )
    _require((folds["status"].astype(str) == "COMPLETED").all(), "D06 contains failed folds")
    _require((folds["task"].astype(str) == "classification").all(), "D06 fold task mismatch")
    _require(not folds.duplicated(["algorithm", "fold"]).any(), "D06 duplicate algorithm/fold rows")
    _require(
        not predictions.duplicated(["algorithm", "row_index"]).any(),
        "D06 duplicate algorithm/row_index predictions",
    )

    folds = folds.copy()
    predictions = predictions.copy()
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
        folds[column] = _as_integer_series(folds, column)
    predictions["fold"] = _as_integer_series(predictions, "fold")
    predictions["row_index"] = _as_integer_series(predictions, "row_index")
    _require((predictions["row_index"] >= 0).all(), "D06 row indexes must be non-negative")

    for column in ("test_used_for_selection", "calibration_test_used_for_selection"):
        folds[column] = folds[column].map(lambda value, field=column: _as_bool(value, field))
    _require(not folds["test_used_for_selection"].any(), "outer test used for threshold selection")
    _require(
        folds["calibration_test_used_for_selection"].all(),
        "D06 calibration suffix is not marked as the threshold-selection sample",
    )

    canonical_windows: list[tuple[int, int]] | None = None
    audit_rows: list[dict[str, Any]] = []
    enriched_groups: list[pd.DataFrame] = []
    checks: list[dict[str, Any]] = []

    def passed(name: str, detail: str) -> None:
        checks.append({"name": name, "passed": True, "detail": detail})

    passed(
        "d04_manifest_and_target",
        "D04 hashes, row schema, execution-return formula and threshold labels verified",
    )
    passed(
        "d06_manifest_and_summary",
        "All D06 output hashes, byte counts and completion declarations verified",
    )

    for algorithm in algorithms:
        algorithm_folds = folds.loc[folds["algorithm"].astype(str) == algorithm].sort_values("fold")
        _require(
            algorithm_folds["fold"].tolist() == list(range(1, fold_count + 1)),
            f"{algorithm} fold identifiers are incomplete",
        )
        windows: list[tuple[int, int]] = []
        for fold_row in algorithm_folds.itertuples(index=False):
            train_start = int(fold_row.train_start_index)
            train_end = int(fold_row.train_end_index)
            test_start = int(fold_row.test_start_index)
            test_end = int(fold_row.test_end_index)
            calibration_start = int(fold_row.calibration_start_index)
            calibration_end = int(fold_row.calibration_end_index)
            calibration_samples = int(fold_row.calibration_samples)
            _require(
                train_start == 0
                and train_end == test_start
                and test_start >= 1
                and test_end > test_start
                and test_end <= len(frame),
                f"{algorithm} fold {fold_row.fold} has invalid expanding-window boundaries",
            )
            _require(
                calibration_start >= train_start
                and calibration_end == train_end
                and calibration_samples == calibration_end - calibration_start
                and calibration_samples >= 2,
                f"{algorithm} fold {fold_row.fold} has invalid calibration boundaries",
            )
            target_labels = frame["target_execution_direction"].astype(str).to_numpy()
            _require(
                set(target_labels[train_start:train_end].tolist()) == set(EXPECTED_CLASSES),
                f"{algorithm} fold {fold_row.fold} training prefix omits a class",
            )
            _require(
                set(target_labels[calibration_start:calibration_end].tolist())
                == set(EXPECTED_CLASSES),
                f"{algorithm} fold {fold_row.fold} calibration suffix omits a class",
            )
            _require(
                set(target_labels[test_start:test_end].tolist()) == set(EXPECTED_CLASSES),
                f"{algorithm} fold {fold_row.fold} outer test omits a class",
            )
            _require(
                str(fold_row.threshold_objective) == "macro_f1",
                f"{algorithm} fold {fold_row.fold} threshold objective mismatch",
            )
            expected_fold = expected_fold_map[int(fold_row.fold)]
            _require(
                (train_start, train_end, test_start, test_end)
                == (
                    expected_fold.train_start,
                    expected_fold.train_end,
                    expected_fold.test_start,
                    expected_fold.test_end,
                ),
                f"{algorithm} fold {fold_row.fold} differs from D06 expanding-fold configuration",
            )
            _require(
                int(fold_row.training_samples) == train_end - train_start
                and int(fold_row.test_samples) == test_end - test_start,
                f"{algorithm} fold {fold_row.fold} sample counts do not match boundaries",
            )
            _require(
                str(fold_row.test_start_date) == dates[test_start],
                f"{algorithm} fold {fold_row.fold} test start date mismatch",
            )
            _require(
                str(fold_row.test_last_feature_date) == dates[test_end - 1],
                f"{algorithm} fold {fold_row.fold} test end date mismatch",
            )
            windows.append((test_start, test_end))

            selected = predictions.loc[
                (predictions["algorithm"].astype(str) == algorithm)
                & (predictions["fold"] == int(fold_row.fold))
            ].sort_values("row_index")
            expected_indices = np.arange(test_start, test_end, dtype=np.int64)
            observed_indices = selected["row_index"].to_numpy(dtype=np.int64)
            _require(
                np.array_equal(observed_indices, expected_indices),
                f"{algorithm} fold {fold_row.fold} OOS row coverage is incomplete",
            )

            expected_dates = np.asarray(dates, dtype=str)[expected_indices]
            observed_dates = selected["feature_date"].astype(str).to_numpy()
            _require(
                np.array_equal(observed_dates, expected_dates),
                f"{algorithm} fold {fold_row.fold} feature dates are misaligned",
            )
            actual = (
                frame["target_execution_direction"].iloc[expected_indices].astype(str).to_numpy()
            )
            stored_actual = selected["actual"].astype(str).to_numpy()
            _require(
                np.array_equal(stored_actual, actual),
                f"{algorithm} fold {fold_row.fold} actual labels differ from D04",
            )

            probability_columns = [
                "probability_down",
                "probability_neutral",
                "probability_up",
            ]
            probabilities = (
                selected[probability_columns]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .to_numpy(dtype=np.float64)
            )
            _require(
                np.all(np.isfinite(probabilities))
                and np.all(probabilities >= 0.0)
                and np.all(probabilities <= 1.0)
                and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
                f"{algorithm} fold {fold_row.fold} probabilities are invalid",
            )
            threshold = float(fold_row.selected_neutral_threshold)
            _require(
                np.isfinite(threshold)
                and 0.0 <= threshold <= 1.0
                and np.any(np.isclose(threshold_grid, threshold, rtol=0.0, atol=1e-12)),
                (
                    f"{algorithm} fold {fold_row.fold} neutral threshold is invalid "
                    "or outside D06 grid"
                ),
            )
            _require(
                np.allclose(
                    pd.to_numeric(selected["selected_neutral_threshold"]).to_numpy(dtype=float),
                    threshold,
                    rtol=0.0,
                    atol=1e-12,
                ),
                f"{algorithm} fold {fold_row.fold} prediction thresholds mismatch",
            )

            argmax = np.asarray(EXPECTED_CLASSES, dtype=str)[np.argmax(probabilities, axis=1)]
            policy = np.asarray(
                apply_neutral_threshold(probabilities, threshold),
                dtype=str,
            )
            _require(
                np.array_equal(selected["argmax_prediction"].astype(str).to_numpy(), argmax),
                f"{algorithm} fold {fold_row.fold} argmax decisions cannot be reproduced",
            )
            _require(
                np.array_equal(selected["policy_prediction"].astype(str).to_numpy(), policy),
                f"{algorithm} fold {fold_row.fold} policy decisions cannot be reproduced",
            )

            recomputed = _classification_metrics("argmax", actual, argmax)
            recomputed.update(_classification_metrics("policy", actual, policy))
            for metric_name, metric_value in recomputed.items():
                _close(
                    getattr(fold_row, metric_name, None),
                    metric_value,
                    f"{algorithm} fold {fold_row.fold} {metric_name}",
                )

            d04_rows = frame.iloc[expected_indices]
            enriched = selected.copy()
            enriched["entry_date"] = d04_rows["entry_date"].astype(str).to_numpy()
            enriched["entry_open"] = pd.to_numeric(d04_rows["entry_open"]).to_numpy(dtype=float)
            enriched["exit_close"] = pd.to_numeric(d04_rows["exit_close"]).to_numpy(dtype=float)
            enriched["actual_execution_return"] = pd.to_numeric(
                d04_rows["target_execution_return"]
            ).to_numpy(dtype=float)
            enriched["argmax_correct"] = argmax == actual
            enriched["policy_correct"] = policy == actual
            enriched["maximum_probability"] = probabilities.max(axis=1)
            sorted_probabilities = np.sort(probabilities, axis=1)
            enriched["probability_margin"] = (
                sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
            )
            enriched_groups.append(enriched)

            audit_row: dict[str, Any] = {
                "algorithm": algorithm,
                "fold": int(fold_row.fold),
                "train_start_index": train_start,
                "train_end_index": train_end,
                "calibration_start_index": calibration_start,
                "calibration_end_index": calibration_end,
                "test_start_index": test_start,
                "test_end_index": test_end,
                "test_samples": len(selected),
                "selected_neutral_threshold": threshold,
                "first_feature_date": expected_dates[0],
                "last_feature_date": expected_dates[-1],
            }
            audit_row.update(recomputed)
            audit_rows.append(audit_row)

        for previous, current in zip(windows, windows[1:], strict=False):
            _require(current[0] >= previous[1], f"{algorithm} outer test windows overlap")
        if canonical_windows is None:
            canonical_windows = windows
        else:
            _require(
                windows == canonical_windows,
                "D06 algorithms do not share identical outer test windows",
            )

    _require(
        len(enriched_groups) == len(algorithms) * fold_count,
        "D07.1 did not audit every algorithm/fold pair",
    )
    _require(
        sum(len(group) for group in enriched_groups) == expected_prediction_rows,
        "D07.1 audited prediction count differs from D06 summary",
    )
    passed("oos_row_alignment", "Every OOS index/date/label is aligned to the same D04 row")
    passed(
        "probability_and_policy_reproduction",
        "All probability vectors and both decision rules were independently reproduced",
    )
    passed(
        "fold_metric_recalculation",
        "Every persisted argmax/policy fold metric was recomputed from row-level predictions",
    )

    verified_aggregate_count = _verify_aggregate_metrics(folds, aggregates, algorithms)
    passed(
        "aggregate_metric_recalculation",
        f"{verified_aggregate_count} D06 aggregate metric rows were independently recomputed",
    )
    passed(
        "temporal_isolation",
        (
            "Calibration suffixes are inside training prefixes; outer test windows "
            "are contiguous, ordered and unused for selection"
        ),
    )

    audited = (
        pd.concat(enriched_groups, ignore_index=True)
        .sort_values(["algorithm", "fold", "row_index"])
        .reset_index(drop=True)
    )
    ticker = str(d04_manifest.get("experiment", {}).get("ticker", "UNKNOWN"))
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]+", "_", ticker).lower()
    run_id = str(d06_summary.get("run_id", policy_run_dir.name))
    safe_run_id = re.sub(r"[^A-Za-z0-9_-]+", "_", run_id).lower()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = output_root / f"{safe_ticker}_{safe_run_id}_audit_{timestamp}"
    audit_dir.mkdir(parents=False, exist_ok=False)

    predictions_path = audit_dir / "audited_predictions.csv"
    folds_path = audit_dir / "fold_audit.csv"
    summary_path = audit_dir / "audit_summary.json"
    atomic_write_csv(predictions_path, audited)
    atomic_write_csv(folds_path, pd.DataFrame(audit_rows))

    returns = audited["actual_execution_return"].to_numpy(dtype=np.float64)
    summary = {
        "schema_version": "2.0",
        "status": "COMPLETED",
        "audit_type": "D07.1_D06_prediction_integrity",
        "created_at": datetime.now(UTC).isoformat(),
        "ticker": ticker,
        "decision_policy_run_id": policy_run_dir.name,
        "execution_dataset": Path(dataset_path).name,
        "execution_dataset_sha256": dataset_hash,
        "source_manifest_sha256": source_manifest_hash,
        "d06_manifest_sha256": sha256_file(policy_run_dir / "decision_policy_manifest.json"),
        "rows_audited": len(audited),
        "algorithm_count": len(algorithms),
        "algorithms": list(algorithms),
        "fold_count": fold_count,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "checks": checks,
        "realized_execution_return_descriptive_only": {
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "minimum": float(np.min(returns)),
            "maximum": float(np.max(returns)),
            "note": "Descriptive target outcomes only; no positions or PnL are applied in D07.1.",
        },
        "limitations": [
            "D07.1 is an integrity audit, not a backtest or profitability assessment.",
            (
                "D07.2 must use this audited output and apply separately declared "
                "execution-cost assumptions."
            ),
            (
                "Historical OOS folds have already been inspected and are not "
                "a prospective untouched test."
            ),
            (
                "SHA-256 manifests detect file changes but are unsigned and do not "
                "authenticate artifact origin."
            ),
        ],
        "output_directory": str(audit_dir),
        "outputs": {
            "predictions": predictions_path.name,
            "fold_audit": folds_path.name,
            "summary": summary_path.name,
            "manifest": "audit_manifest.json",
        },
    }
    atomic_write_json(summary_path, summary)

    output_paths = (predictions_path, folds_path, summary_path)
    audit_manifest = {
        "schema_version": "2.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": {
            "decision_policy_run_id": policy_run_dir.name,
            "d06_manifest_sha256": sha256_file(policy_run_dir / "decision_policy_manifest.json"),
            "execution_dataset": Path(dataset_path).name,
            "execution_dataset_sha256": dataset_hash,
            "source_manifest_sha256": source_manifest_hash,
        },
        "outputs": manifest_output_entries(output_paths),
    }
    atomic_write_json(audit_dir / "audit_manifest.json", audit_manifest)
    return summary
