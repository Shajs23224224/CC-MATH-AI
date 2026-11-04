"""D07.1 regression tests against production-shaped D04/D06 artifacts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from execution.research.d07_audit import audit_decision_policy_run
from tests.unit.d07_fixture_factory import (
    _manifest_entry,
    _sha256,
    _write_json,
    build_d04_d06_artifacts,
)


def _refresh_d06_manifest(policy_dir, filename):
    manifest_path = policy_dir / "decision_policy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][filename] = _manifest_entry(policy_dir / filename)
    _write_json(manifest_path, manifest)


def test_audit_reconciles_d06_predictions_with_d04_and_writes_manifest(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    result = audit_decision_policy_run(
        artifacts["policy_dir"],
        artifacts["dataset_path"],
        output_root=tmp_path / "audits",
    )

    assert result["status"] == "COMPLETED"
    assert result["rows_audited"] == 6
    assert result["algorithm_count"] == 1
    assert result["fold_count"] == 1
    assert result["checks_passed"] == result["checks_total"]
    assert result["checks_total"] >= 7

    audit_dir = tmp_path / "audits" / result["output_directory"].split("/")[-1]
    audited = pd.read_csv(audit_dir / "audited_predictions.csv")
    assert len(audited) == 6
    assert {
        "entry_date",
        "entry_open",
        "exit_close",
        "actual_execution_return",
        "argmax_correct",
        "policy_correct",
        "probability_margin",
    }.issubset(audited.columns)
    assert audited.loc[0, "argmax_prediction"] == "DOWN"
    assert audited.loc[0, "policy_prediction"] == "NEUTRAL"
    assert audited.loc[0, "row_index"] == 12
    assert audited.loc[0, "actual_execution_return"] == pytest.approx(-0.012)

    manifest = json.loads((audit_dir / "audit_manifest.json").read_text(encoding="utf-8"))
    for filename, entry in manifest["outputs"].items():
        path = audit_dir / filename
        assert path.is_file()
        assert entry["bytes"] == path.stat().st_size
        assert len(entry["sha256"]) == 64


def test_audit_rejects_d06_file_tampering_before_reconciliation(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    with artifacts["predictions_path"].open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(
        ValueError,
        match="byte-count verification failed|SHA-256 verification failed",
    ):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_prediction_row_index_misalignment(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    predictions = pd.read_csv(artifacts["predictions_path"])
    predictions.loc[0, "row_index"] = 5
    predictions.to_csv(
        artifacts["predictions_path"],
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_d06_manifest(artifacts["policy_dir"], "oos_predictions.csv")

    with pytest.raises(ValueError, match="OOS row coverage is incomplete"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_recomputes_policy_predictions_instead_of_trusting_csv(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    predictions = pd.read_csv(artifacts["predictions_path"])
    predictions.loc[0, "policy_prediction"] = "UP"
    predictions.to_csv(
        artifacts["predictions_path"],
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_d06_manifest(artifacts["policy_dir"], "oos_predictions.csv")

    with pytest.raises(ValueError, match="policy decisions cannot be reproduced"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_outer_test_selection_leakage(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    folds = pd.read_csv(artifacts["fold_metrics_path"])
    folds.loc[0, "test_used_for_selection"] = True
    folds.to_csv(
        artifacts["fold_metrics_path"],
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_d06_manifest(artifacts["policy_dir"], "fold_metrics.csv")

    with pytest.raises(ValueError, match="outer test used for threshold selection"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_recomputes_aggregate_metrics(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    aggregates = pd.read_csv(artifacts["aggregate_metrics_path"])
    aggregates.loc[0, "mean"] = float(aggregates.loc[0, "mean"]) + 0.10
    aggregates.to_csv(
        artifacts["aggregate_metrics_path"],
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_d06_manifest(artifacts["policy_dir"], "aggregate_metrics.csv")

    with pytest.raises(ValueError, match="aggregate .* mismatch"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_aggregate_task_mismatch(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    aggregates = pd.read_csv(artifacts["aggregate_metrics_path"])
    aggregates.loc[0, "task"] = "regression"
    aggregates.to_csv(
        artifacts["aggregate_metrics_path"],
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_d06_manifest(artifacts["policy_dir"], "aggregate_metrics.csv")

    with pytest.raises(ValueError, match="aggregate task must be classification"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_reconstructs_d06_fold_geometry_from_manifest(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    manifest_path = artifacts["policy_dir"] / "decision_policy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["test_fraction"] = 0.30
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="differs from D06 expanding-fold configuration"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_reconstructs_exact_d06_calibration_suffix(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    folds = pd.read_csv(artifacts["fold_metrics_path"])
    folds.loc[0, "calibration_start_index"] = 8
    folds.loc[0, "calibration_samples"] = 4
    folds.to_csv(
        artifacts["fold_metrics_path"],
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_d06_manifest(artifacts["policy_dir"], "fold_metrics.csv")

    with pytest.raises(ValueError, match="calibration window differs from D06 configuration"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_d04_threshold_declared_as_transaction_cost(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    d04_manifest_path = artifacts["dataset_manifest_path"]
    d04_manifest = json.loads(d04_manifest_path.read_text(encoding="utf-8"))
    d04_manifest["experiment"]["threshold_is_transaction_cost"] = True
    _write_json(d04_manifest_path, d04_manifest)

    policy_manifest_path = artifacts["policy_dir"] / "decision_policy_manifest.json"
    policy_manifest = json.loads(policy_manifest_path.read_text(encoding="utf-8"))
    policy_manifest["input"]["source_manifest_sha256"] = _sha256(d04_manifest_path)
    _write_json(policy_manifest_path, policy_manifest)

    with pytest.raises(ValueError, match="must be a label band"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_entry_date_not_next_observed_session(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    dataset_path = artifacts["dataset_path"]
    dataset = pd.read_csv(dataset_path)
    dataset.loc[0, "entry_date"] = "2024-01-05"
    dataset.to_csv(
        dataset_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    d04_manifest_path = artifacts["dataset_manifest_path"]
    d04_manifest = json.loads(d04_manifest_path.read_text(encoding="utf-8"))
    d04_manifest["artifact"].update(_manifest_entry(dataset_path))
    _write_json(d04_manifest_path, d04_manifest)

    policy_dir = artifacts["policy_dir"]
    policy_manifest_path = policy_dir / "decision_policy_manifest.json"
    policy_manifest = json.loads(policy_manifest_path.read_text(encoding="utf-8"))
    policy_manifest["input"]["execution_dataset_sha256"] = _sha256(dataset_path)
    policy_manifest["input"]["source_manifest_sha256"] = _sha256(d04_manifest_path)
    summary_path = policy_dir / "decision_policy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["snapshot_sha256"] = _sha256(dataset_path)
    _write_json(summary_path, summary)
    policy_manifest["outputs"]["decision_policy_summary.json"] = _manifest_entry(
        summary_path
    )
    _write_json(policy_manifest_path, policy_manifest)

    with pytest.raises(ValueError, match="next observed feature session"):
        audit_decision_policy_run(
            policy_dir,
            dataset_path,
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_changed_d04_dataset(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    dataset = pd.read_csv(artifacts["dataset_path"])
    dataset.loc[0, "entry_open"] = 101.0
    dataset.to_csv(
        artifacts["dataset_path"],
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    with pytest.raises(ValueError, match="D04 dataset SHA-256 mismatch"):
        audit_decision_policy_run(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )
