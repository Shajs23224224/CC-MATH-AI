"""Synthetic D04/D06 artifacts that preserve the production file contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_execution_decision_policy import (
    _aggregate_metrics,
    _policy_metrics,
    apply_neutral_threshold,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _manifest_entry(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def build_d04_d06_artifacts(root: Path) -> dict[str, Path]:
    """Create one completed D04 dataset and one completed D06 decision-policy run."""
    root.mkdir(parents=True, exist_ok=True)
    feature_dates = pd.bdate_range("2024-01-02", periods=12)
    returns = np.asarray(
        [
            -0.020,
            0.003,
            0.005,
            0.015,
            -0.010,
            0.0004,
            -0.020,
            0.005,
            0.0001,
            -0.020,
            0.020,
            0.0005,
        ],
        dtype=np.float64,
    )
    threshold = 0.001
    labels = np.where(
        returns > threshold,
        "UP",
        np.where(returns < -threshold, "DOWN", "NEUTRAL"),
    )
    entry_dates = feature_dates + pd.offsets.BDay(1)
    entry_open = np.full(12, 100.0, dtype=np.float64)
    exit_close = entry_open * (1.0 + returns)
    dataset = pd.DataFrame(
        {
            "feature_a": np.arange(12, dtype=np.float64),
            "feature_b": np.asarray([0.1, 0.2, 0.3] * 4, dtype=np.float64),
            "entry_date": entry_dates.strftime("%Y-%m-%d"),
            "entry_open": entry_open,
            "exit_close": exit_close,
            "target_execution_return": returns,
            "target_execution_direction": labels,
        },
        index=feature_dates,
    )
    dataset.index.name = "feature_date"
    d04_dir = root / "d04_execution"
    d04_dir.mkdir()
    dataset_path = d04_dir / "execution_dataset.csv"
    dataset.to_csv(
        dataset_path,
        index_label="feature_date",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
        lineterminator="\n",
    )
    d04_manifest = {
        "schema_version": "1.0",
        "experiment": {
            "ticker": "TEST",
            "threshold_bps": 10.0,
            "threshold_fraction": threshold,
            "threshold_is_transaction_cost": False,
        },
        "target": {
            "classification_name": "target_execution_direction",
            "classes": ["DOWN", "NEUTRAL", "UP"],
        },
        "dataset": {
            "rows": len(dataset),
            "feature_names": ["feature_a", "feature_b"],
        },
        "artifact": {
            "file": dataset_path.name,
            "sha256": _sha256(dataset_path),
            "bytes": dataset_path.stat().st_size,
        },
    }
    source_manifest_path = d04_dir / "execution_dataset_manifest.json"
    _write_json(source_manifest_path, d04_manifest)

    policy_dir = root / "d06_decision_policy"
    policy_dir.mkdir()
    test_indices = np.arange(6, 12, dtype=np.int64)
    actual = labels[test_indices].astype(str)
    probabilities = np.asarray(
        [
            [0.45, 0.40, 0.15],
            [0.10, 0.15, 0.75],
            [0.10, 0.70, 0.20],
            [0.65, 0.20, 0.15],
            [0.15, 0.20, 0.65],
            [0.20, 0.60, 0.20],
        ],
        dtype=np.float64,
    )
    neutral_threshold = 0.40
    argmax = np.asarray(("DOWN", "UP", "NEUTRAL", "DOWN", "UP", "NEUTRAL"))
    policy = np.asarray(
        apply_neutral_threshold(probabilities, neutral_threshold),
        dtype=str,
    )
    prediction_rows = pd.DataFrame(
        {
            "fold": np.ones(len(test_indices), dtype=np.int64),
            "algorithm": ["test_classifier"] * len(test_indices),
            "row_index": test_indices,
            "feature_date": [value.isoformat() for value in feature_dates[test_indices]],
            "actual": actual,
            "argmax_prediction": argmax,
            "policy_prediction": policy,
            "selected_neutral_threshold": neutral_threshold,
            "probability_down": probabilities[:, 0],
            "probability_neutral": probabilities[:, 1],
            "probability_up": probabilities[:, 2],
        }
    )
    predictions_path = policy_dir / "oos_predictions.csv"
    prediction_rows.to_csv(
        predictions_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    fold_record: dict[str, Any] = {
        "fold": 1,
        "task": "classification",
        "algorithm": "test_classifier",
        "status": "COMPLETED",
        "train_start_index": 0,
        "train_end_index": 6,
        "test_start_index": 6,
        "test_end_index": 12,
        "training_samples": 6,
        "test_samples": 6,
        "test_used_for_selection": False,
        "test_start_date": feature_dates[6].isoformat(),
        "test_last_feature_date": feature_dates[11].isoformat(),
        "calibration_start_index": 3,
        "calibration_end_index": 6,
        "calibration_samples": 3,
        "calibration_test_used_for_selection": True,
        "threshold_objective": "macro_f1",
        "selected_neutral_threshold": neutral_threshold,
        "calibration_macro_f1": 0.5,
        "calibration_balanced_accuracy": 0.5,
        "calibration_neutral_precision": 0.5,
        "calibration_neutral_recall": 0.5,
    }
    fold_record.update(_policy_metrics("argmax", actual, argmax))
    fold_record.update(_policy_metrics("policy", actual, policy))
    fold_path = policy_dir / "fold_metrics.csv"
    pd.DataFrame([fold_record]).to_csv(
        fold_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    aggregate_path = policy_dir / "aggregate_metrics.csv"
    pd.DataFrame(_aggregate_metrics([fold_record])).to_csv(
        aggregate_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    summary = {
        "run_id": policy_dir.name,
        "created_at": "2024-01-01T00:00:00+00:00",
        "ticker": "TEST",
        "status": "COMPLETED",
        "algorithms": ["test_classifier"],
        "fold_count": 1,
        "attempted_model_folds": 1,
        "completed_model_folds": 1,
        "failed_model_folds": 0,
        "prediction_rows": len(prediction_rows),
        "classes": ["DOWN", "NEUTRAL", "UP"],
        "threshold_grid": [0.40],
        "threshold_selection_objective": "macro_f1",
        "calibration_fraction": 0.20,
        "outer_test_used_for_selection": False,
        "snapshot_file": dataset_path.name,
        "snapshot_sha256": _sha256(dataset_path),
        "outputs": {
            "fold_metrics": fold_path.name,
            "aggregate_metrics": aggregate_path.name,
            "predictions": predictions_path.name,
        },
    }
    summary_path = policy_dir / "decision_policy_summary.json"
    _write_json(summary_path, summary)
    policy_manifest = {
        "schema_version": "1.0",
        "input": {
            "execution_dataset": dataset_path.name,
            "execution_dataset_sha256": _sha256(dataset_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
        },
        "configuration": {
            "seed": 28,
            "fold_count": 1,
            "test_fraction": 0.10,
            "calibration_fraction": 0.20,
            "algorithms": ["test_classifier"],
            "threshold_grid": [0.40],
            "selection_objective": "macro_f1",
        },
        "outputs": {
            path.name: _manifest_entry(path)
            for path in (
                fold_path,
                aggregate_path,
                predictions_path,
                summary_path,
            )
        },
    }
    _write_json(policy_dir / "decision_policy_manifest.json", policy_manifest)
    return {
        "dataset_path": dataset_path,
        "dataset_manifest_path": source_manifest_path,
        "policy_dir": policy_dir,
        "predictions_path": predictions_path,
        "fold_metrics_path": fold_path,
        "aggregate_metrics_path": aggregate_path,
    }
