"""Tests for D07.1 integrity auditing of D06 execution-policy predictions."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.audit_execution_decision_policy import (
    _apply_neutral_threshold,
    _classification_metrics,
    audit_decision_policy_run,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _refresh_policy_output_hash(run_dir, filename):
    manifest_path = run_dir / "decision_policy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][filename] = {
        "sha256": _sha256(run_dir / filename),
        "bytes": (run_dir / filename).stat().st_size,
    }
    _write_json(manifest_path, manifest)


def create_audit_fixture(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=12)
    execution_returns = np.asarray([
        -0.020, 0.003, 0.005, 0.015, -0.010, 0.0004,
        -0.020, 0.005, 0.0001, -0.020, 0.020, 0.0005,
    ])
    threshold = 0.001
    labels = np.where(
        execution_returns > threshold,
        "UP",
        np.where(execution_returns < -threshold, "DOWN", "NEUTRAL"),
    )

    dataset_frame = pd.DataFrame(
        {
            "feature_a": np.arange(12, dtype=float),
            "feature_b": np.arange(12, dtype=float) % 3,
            "entry_date": (dates + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
            "entry_open": np.full(12, 100.0),
            "exit_close": 100.0 * (1.0 + execution_returns),
            "target_execution_return": execution_returns,
            "target_execution_direction": labels,
        },
        index=dates,
    )
    dataset_frame.index.name = "feature_date"

    dataset_path = tmp_path / "execution_dataset.csv"
    dataset_frame.to_csv(
        dataset_path,
        index_label="feature_date",
        float_format="%.17g",
    )

    dataset_manifest = {
        "experiment": {
            "ticker": "TEST",
            "threshold_bps": 10.0,
            "threshold_fraction": threshold,
        },
        "artifact": {
            "file": dataset_path.name,
            "sha256": _sha256(dataset_path),
        },
        "dataset": {
            "rows": len(dataset_frame),
            "feature_names": ["feature_a", "feature_b"],
        },
        "target": {
            "classification_name": "target_execution_direction",
        },
    }
    source_manifest_path = tmp_path / "execution_dataset_manifest.json"
    _write_json(source_manifest_path, dataset_manifest)

    run_dir = tmp_path / "test_decision_policy_run"
    run_dir.mkdir()

    actual = labels[6:12].astype(str)
    probabilities = np.asarray([
        [0.75, 0.15, 0.10],
        [0.65, 0.15, 0.20],
        [0.20, 0.55, 0.25],
        [0.60, 0.25, 0.15],
        [0.20, 0.35, 0.45],
        [0.20, 0.45, 0.35],
    ])
    threshold_selected = 0.40
    argmax = np.asarray(("DOWN", "DOWN", "NEUTRAL", "DOWN", "UP", "NEUTRAL"))
    policy = _apply_neutral_threshold(probabilities, threshold_selected)

    prediction_frame = pd.DataFrame(
        {
            "fold": np.ones(6, dtype=int),
            "algorithm": ["test_classifier"] * 6,
            "row_index": np.arange(6, 12, dtype=int),
            "feature_date": [value.isoformat() for value in dates[6:12]],
            "actual": actual,
            "argmax_prediction": argmax,
            "policy_prediction": policy,
            "selected_neutral_threshold": np.full(6, threshold_selected),
            "probability_down": probabilities[:, 0],
            "probability_neutral": probabilities[:, 1],
            "probability_up": probabilities[:, 2],
        }
    )
    prediction_path = run_dir / "oos_predictions.csv"
    prediction_frame.to_csv(
        prediction_path,
        index=False,
        float_format="%.17g",
    )

    fold_record = {
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
        "calibration_start_index": 3,
        "calibration_end_index": 6,
        "calibration_samples": 3,
        "calibration_test_used_for_selection": True,
        "threshold_objective": "macro_f1",
        "selected_neutral_threshold": threshold_selected,
    }
    fold_record.update(_classification_metrics("argmax", actual, argmax))
    fold_record.update(_classification_metrics("policy", actual, policy))

    fold_path = run_dir / "fold_metrics.csv"
    pd.DataFrame([fold_record]).to_csv(
        fold_path,
        index=False,
        float_format="%.17g",
    )

    aggregate_rows = []
    for metric in (
        "argmax_accuracy",
        "policy_accuracy",
        "argmax_balanced_accuracy",
        "policy_balanced_accuracy",
        "argmax_macro_f1",
        "policy_macro_f1",
        "policy_neutral_precision",
        "policy_neutral_recall",
    ):
        value = float(fold_record[metric])
        aggregate_rows.append(
            {
                "task": "classification",
                "algorithm": "test_classifier",
                "metric": metric,
                "successful_folds": 1,
                "mean": value,
                "std": 0.0,
                "min": value,
                "max": value,
            }
        )
    aggregate_path = run_dir / "aggregate_metrics.csv"
    pd.DataFrame(aggregate_rows).to_csv(
        aggregate_path,
        index=False,
        float_format="%.17g",
    )

    summary = {
        "run_id": run_dir.name,
        "status": "COMPLETED",
        "ticker": "TEST",
        "algorithms": ["test_classifier"],
        "fold_count": 1,
        "attempted_model_folds": 1,
        "completed_model_folds": 1,
        "failed_model_folds": 0,
        "prediction_rows": 6,
        "classes": ["DOWN", "NEUTRAL", "UP"],
        "threshold_grid": [0.05, 0.40, 0.80],
        "threshold_selection_objective": "macro_f1",
        "calibration_fraction": 0.20,
        "outer_test_used_for_selection": False,
        "snapshot_file": dataset_path.name,
        "snapshot_sha256": _sha256(dataset_path),
        "outputs": {
            "fold_metrics": fold_path.name,
            "aggregate_metrics": aggregate_path.name,
            "predictions": prediction_path.name,
        },
    }
    summary_path = run_dir / "decision_policy_summary.json"
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
            "threshold_grid": summary["threshold_grid"],
            "selection_objective": "macro_f1",
        },
        "outputs": {
            path.name: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                fold_path,
                aggregate_path,
                prediction_path,
                summary_path,
            )
        },
    }
    _write_json(run_dir / "decision_policy_manifest.json", policy_manifest)
    return run_dir, dataset_path, prediction_path


def test_audit_enriches_predictions_and_writes_verified_outputs(tmp_path):
    run_dir, dataset_path, _ = create_audit_fixture(tmp_path)

    summary = audit_decision_policy_run(
        run_dir,
        dataset_path,
        output_root=tmp_path / "audits",
    )

    assert summary["status"] == "COMPLETED"
    assert summary["audit_type"] == "D07.1_D06_prediction_integrity"
    assert summary["rows_audited"] == 6
    assert summary["checks_passed"] == summary["checks_total"]
    assert summary["per_algorithm"][0]["algorithm"] == "test_classifier"
    assert summary["outputs"]["predictions"] == "audited_predictions.csv"

    output_dir = tmp_path / "audits" / summary["output_directory"].split("/")[-1]
    audited = pd.read_csv(output_dir / "audited_predictions.csv")

    assert len(audited) == 6
    assert "actual_execution_return" in audited.columns
    assert "entry_open" in audited.columns
    assert "exit_close" in audited.columns
    assert "argmax_correct" in audited.columns
    assert "policy_correct" in audited.columns
    assert "probability_margin" in audited.columns
    assert (output_dir / "fold_audit.csv").is_file()
    assert (output_dir / "audit_summary.json").is_file()
    assert (output_dir / "audit_manifest.json").is_file()


def test_audit_rejects_tampered_d06_prediction_artifact(tmp_path):
    run_dir, dataset_path, prediction_path = create_audit_fixture(tmp_path)

    with prediction_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="sha256_oos_predictions.csv"):
        audit_decision_policy_run(
            run_dir,
            dataset_path,
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_policy_prediction_that_cannot_be_reproduced(tmp_path):
    run_dir, dataset_path, prediction_path = create_audit_fixture(tmp_path)
    predictions = pd.read_csv(prediction_path)
    predictions.loc[0, "policy_prediction"] = "UP"
    predictions.to_csv(
        prediction_path,
        index=False,
        float_format="%.17g",
    )
    _refresh_policy_output_hash(run_dir, "oos_predictions.csv")

    with pytest.raises(ValueError, match="policy predictions cannot be reproduced"):
        audit_decision_policy_run(
            run_dir,
            dataset_path,
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_probability_rows_that_do_not_sum_to_one(tmp_path):
    run_dir, dataset_path, prediction_path = create_audit_fixture(tmp_path)
    predictions = pd.read_csv(prediction_path)
    predictions.loc[0, "probability_neutral"] = 0.50
    predictions.to_csv(
        prediction_path,
        index=False,
        float_format="%.17g",
    )
    _refresh_policy_output_hash(run_dir, "oos_predictions.csv")

    with pytest.raises(ValueError, match="probability matrix is invalid"):
        audit_decision_policy_run(
            run_dir,
            dataset_path,
            output_root=tmp_path / "audits",
        )


def test_neutral_threshold_policy_is_deterministic():
    probabilities = np.asarray([
        [0.70, 0.20, 0.10],
        [0.20, 0.50, 0.30],
        [0.10, 0.20, 0.70],
    ])

    assert _apply_neutral_threshold(
        probabilities,
        0.40,
    ).tolist() == ["DOWN", "NEUTRAL", "UP"]

    assert _apply_neutral_threshold(
        probabilities,
        0.60,
    ).tolist() == ["DOWN", "UP", "UP"]
