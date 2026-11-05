"""Regression tests for independent D06.3 temperature and reliability auditing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execution.research.d07_calibration import (
    _apply_temperature,
    _brier_score,
    _fit_temperature_independently,
    _log_loss,
    _neutral_decisions,
    _reliability_diagnostics,
    _select_threshold_independently,
    audit_calibration_reliability,
)
from scripts.run_execution_decision_policy import (
    _aggregate_metrics,
    _policy_metrics,
)
from tests.unit.d07_fixture_factory import (
    _manifest_entry,
    _sha256,
    _write_json,
    build_d04_d06_artifacts,
)


def _build_d06_3_artifacts(root: Path) -> dict[str, Path | float]:
    """Upgrade the small D07 fixture to the persisted D06.3 artifact contract."""
    artifacts = build_d04_d06_artifacts(root)
    policy_dir = artifacts["policy_dir"]
    dataset = pd.read_csv(artifacts["dataset_path"])
    labels = dataset["target_execution_direction"].astype(str).to_numpy()

    calibration_indices = np.arange(9, 12, dtype=np.int64)
    calibration_actual = labels[calibration_indices]
    raw_calibration = np.asarray(
        [
            [0.72, 0.18, 0.10],
            [0.12, 0.18, 0.70],
            [0.18, 0.62, 0.20],
        ],
        dtype=np.float64,
    )
    bounds = (0.05, 20.0)
    temperature, calibration_scores = _fit_temperature_independently(
        calibration_actual,
        raw_calibration,
        bounds,
    )
    scaled_calibration = _apply_temperature(raw_calibration, temperature)
    threshold_result = _select_threshold_independently(
        calibration_actual,
        scaled_calibration,
        (0.40,),
    )

    calibration_predictions_path = policy_dir / "calibration_predictions.csv"
    pd.DataFrame(
        {
            "fold": 1,
            "algorithm": "test_classifier",
            "row_index": calibration_indices,
            "feature_date": pd.to_datetime(dataset.loc[calibration_indices, "feature_date"])
            .dt.strftime("%Y-%m-%dT%H:%M:%S")
            .to_numpy(),
            "actual": calibration_actual,
            "raw_probability_down": raw_calibration[:, 0],
            "raw_probability_neutral": raw_calibration[:, 1],
            "raw_probability_up": raw_calibration[:, 2],
        }
    ).to_csv(
        calibration_predictions_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    predictions_path = artifacts["predictions_path"]
    predictions = pd.read_csv(predictions_path)
    raw_oos = predictions[["probability_down", "probability_neutral", "probability_up"]].to_numpy(
        dtype=np.float64
    )
    scaled_oos = _apply_temperature(raw_oos, temperature)
    actual_oos = predictions["actual"].astype(str).to_numpy()
    predictions["raw_probability_down"] = raw_oos[:, 0]
    predictions["raw_probability_neutral"] = raw_oos[:, 1]
    predictions["raw_probability_up"] = raw_oos[:, 2]
    predictions["probability_down"] = scaled_oos[:, 0]
    predictions["probability_neutral"] = scaled_oos[:, 1]
    predictions["probability_up"] = scaled_oos[:, 2]
    predictions["argmax_prediction"] = np.asarray(["DOWN", "NEUTRAL", "UP"], dtype=str)[
        np.argmax(scaled_oos, axis=1)
    ]
    predictions["policy_prediction"] = _neutral_decisions(
        scaled_oos,
        threshold_result["selected_neutral_threshold"],
    )
    predictions.to_csv(
        predictions_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    reliability_rows, reliability_metrics = _reliability_diagnostics(
        actual_oos,
        scaled_oos,
        algorithm="test_classifier",
        fold=1,
        n_bins=10,
    )
    fold_path = artifacts["fold_metrics_path"]
    folds = pd.read_csv(fold_path)
    fold = folds.iloc[0].to_dict()
    fold.update(
        {
            "probability_calibration_temperature": temperature,
            **calibration_scores,
            **threshold_result,
            "multiclass_log_loss": _log_loss(actual_oos, scaled_oos),
            "multiclass_brier_score": _brier_score(actual_oos, scaled_oos),
            **reliability_metrics,
        }
    )
    fold.update(
        _policy_metrics(
            "argmax",
            actual_oos,
            predictions["argmax_prediction"].astype(str).to_numpy(),
        )
    )
    fold.update(
        _policy_metrics(
            "policy",
            actual_oos,
            predictions["policy_prediction"].astype(str).to_numpy(),
        )
    )
    pd.DataFrame([fold]).to_csv(
        fold_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    aggregate_path = artifacts["aggregate_metrics_path"]
    pd.DataFrame(_aggregate_metrics([fold])).to_csv(
        aggregate_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    summary_path = policy_dir / "decision_policy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "probability_calibration_method": "multiclass_temperature_scaling",
            "probability_calibration_temperature_bounds": list(bounds),
            "reliability_bin_count": 10,
            "reliability_bins": reliability_rows,
        }
    )
    summary["outputs"]["calibration_predictions"] = calibration_predictions_path.name
    _write_json(summary_path, summary)

    manifest_path = policy_dir / "decision_policy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_version"] = "D06.3"
    manifest["configuration"].update(
        {
            "probability_calibration_method": "multiclass_temperature_scaling",
            "probability_calibration_temperature_bounds": list(bounds),
            "reliability_bin_count": 10,
            "classes": ["DOWN", "NEUTRAL", "UP"],
        }
    )
    manifest["outputs"] = {
        path.name: _manifest_entry(path)
        for path in (
            fold_path,
            aggregate_path,
            predictions_path,
            summary_path,
            calibration_predictions_path,
        )
    }
    _write_json(manifest_path, manifest)

    return {
        "dataset_path": artifacts["dataset_path"],
        "policy_dir": policy_dir,
        "temperature": temperature,
        "predictions_path": predictions_path,
        "calibration_predictions_path": calibration_predictions_path,
        "fold_metrics_path": fold_path,
    }


def _refresh_manifest(policy_dir: Path, filename: str) -> None:
    path = policy_dir / "decision_policy_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["outputs"][filename] = _manifest_entry(policy_dir / filename)
    _write_json(path, manifest)


def test_audit_reproduces_temperature_threshold_and_reliability_bins(tmp_path):
    artifacts = _build_d06_3_artifacts(tmp_path)
    result = audit_calibration_reliability(
        artifacts["policy_dir"],
        artifacts["dataset_path"],
        output_root=tmp_path / "calibration_audits",
    )

    assert result["status"] == "COMPLETED"
    assert result["audit_type"] == "D07.3_D06_calibration_and_reliability"
    assert result["checks_passed"] == result["checks_total"] == 7
    assert result["calibration_rows_audited"] == 3
    assert result["oos_rows_audited"] == 6
    assert result["reliability_bins_audited"] == 30

    audit_dir = Path(result["output_directory"])
    fold_audit = pd.read_csv(audit_dir / "calibration_fold_audit.csv")
    bins_audit = pd.read_csv(audit_dir / "reliability_bins_audit.csv")
    assert fold_audit.loc[0, "recomputed_temperature"] == pytest.approx(
        artifacts["temperature"], rel=1e-7, abs=1e-9
    )
    assert fold_audit.loc[0, "recomputed_threshold"] == pytest.approx(0.40)
    assert len(bins_audit) == 30
    assert bins_audit["passed"].all()

    manifest = json.loads(
        (audit_dir / "calibration_audit_manifest.json").read_text(encoding="utf-8")
    )
    for filename, entry in manifest["outputs"].items():
        path = audit_dir / filename
        assert path.is_file()
        assert entry["sha256"] == _sha256(path)
        assert entry["bytes"] == path.stat().st_size


def test_audit_rejects_tampered_reliability_bin_after_manifest_refresh(tmp_path):
    artifacts = _build_d06_3_artifacts(tmp_path)
    summary_path = Path(artifacts["policy_dir"]) / "decision_policy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["reliability_bins"][0]["sample_count"] += 1
    _write_json(summary_path, summary)
    _refresh_manifest(Path(artifacts["policy_dir"]), "decision_policy_summary.json")

    with pytest.raises(ValueError, match="sample_count mismatch"):
        audit_calibration_reliability(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_raw_oos_probability_tampering_after_manifest_refresh(tmp_path):
    artifacts = _build_d06_3_artifacts(tmp_path)
    path = Path(artifacts["predictions_path"])
    predictions = pd.read_csv(path)
    predictions.loc[0, "raw_probability_down"] += 0.05
    predictions.loc[0, "raw_probability_neutral"] -= 0.05
    predictions.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_manifest(Path(artifacts["policy_dir"]), "oos_predictions.csv")

    with pytest.raises(ValueError, match="OOS probabilities do not match"):
        audit_calibration_reliability(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_modified_temperature_after_manifest_refresh(tmp_path):
    artifacts = _build_d06_3_artifacts(tmp_path)
    path = Path(artifacts["fold_metrics_path"])
    folds = pd.read_csv(path)
    folds.loc[0, "probability_calibration_temperature"] *= 1.5
    folds.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_manifest(Path(artifacts["policy_dir"]), "fold_metrics.csv")

    with pytest.raises(ValueError, match="independently fitted temperature mismatch"):
        audit_calibration_reliability(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_d04_next_session_mapping_even_after_hash_refresh(tmp_path):
    artifacts = _build_d06_3_artifacts(tmp_path)
    dataset_path = Path(artifacts["dataset_path"])
    dataset = pd.read_csv(dataset_path, index_col="feature_date")
    dataset.iloc[0, dataset.columns.get_loc("entry_date")] = "2024-01-05"
    dataset.to_csv(
        dataset_path,
        index_label="feature_date",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
        lineterminator="\n",
    )

    d04_manifest_path = dataset_path.parent / "execution_dataset_manifest.json"
    d04_manifest = json.loads(d04_manifest_path.read_text(encoding="utf-8"))
    d04_manifest["artifact"]["sha256"] = _sha256(dataset_path)
    d04_manifest["artifact"]["bytes"] = dataset_path.stat().st_size
    _write_json(d04_manifest_path, d04_manifest)

    policy_dir = Path(artifacts["policy_dir"])
    d06_manifest_path = policy_dir / "decision_policy_manifest.json"
    d06_manifest = json.loads(d06_manifest_path.read_text(encoding="utf-8"))
    d06_manifest["input"]["execution_dataset_sha256"] = _sha256(dataset_path)
    d06_manifest["input"]["source_manifest_sha256"] = _sha256(d04_manifest_path)
    _write_json(d06_manifest_path, d06_manifest)
    summary_path = policy_dir / "decision_policy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["snapshot_sha256"] = _sha256(dataset_path)
    _write_json(summary_path, summary)
    _refresh_manifest(policy_dir, "decision_policy_summary.json")

    with pytest.raises(
        ValueError,
        match="D04 entry dates do not map to the next observed feature session",
    ):
        audit_calibration_reliability(
            policy_dir,
            dataset_path,
            output_root=tmp_path / "audits",
        )


def test_audit_rejects_calibration_label_misalignment_after_manifest_refresh(tmp_path):
    artifacts = _build_d06_3_artifacts(tmp_path)
    path = Path(artifacts["calibration_predictions_path"])
    calibration = pd.read_csv(path)
    calibration.loc[0, "actual"] = "UP"
    calibration.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    _refresh_manifest(Path(artifacts["policy_dir"]), "calibration_predictions.csv")

    with pytest.raises(ValueError, match="calibration labels differ from D04"):
        audit_calibration_reliability(
            artifacts["policy_dir"],
            artifacts["dataset_path"],
            output_root=tmp_path / "audits",
        )
