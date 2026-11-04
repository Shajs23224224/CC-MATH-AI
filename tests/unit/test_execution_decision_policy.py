"""D06 tests: nested temporal threshold selection and auditable OOS outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.contracts.ml import MLDataset
from scripts.build_execution_dataset import build_execution_dataset
from scripts.run_execution_decision_policy import (
    apply_neutral_threshold,
    apply_temperature_scaling,
    choose_neutral_threshold,
    evaluate_decision_policy_fold,
    fit_temperature_scaling,
    run_decision_policy,
)
from scripts.run_ml_walk_forward import build_expanding_folds


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification_dataset(size=120):
    features = tuple(
        (float(index), float(index % 3), float((index % 7) / 7.0)) for index in range(size)
    )
    labels = tuple(("DOWN", "NEUTRAL", "UP")[index % 3] for index in range(size))
    return MLDataset(
        dataset_id="decision-policy-synthetic",
        source="synthetic chronological test fixture",
        feature_names=("trend", "cycle", "range"),
        target_name="direction",
        features=features,
        target=labels,
        task="classification",
    )


def create_d04_dataset(tmp_path, size=180):
    dates = pd.bdate_range("2020-01-02", periods=size + 1)
    closes = np.asarray(
        [100.0] + [value for _ in range((size + 2) // 3) for value in (100.5, 99.5, 100.0)],
        dtype=np.float64,
    )[: size + 1]
    raw = pd.DataFrame(
        {
            "open": np.full(size + 1, 100.0),
            "high": np.maximum(closes, 100.0) + 0.2,
            "low": np.minimum(closes, 100.0) - 0.2,
            "close": closes,
            "volume": np.full(size + 1, 250_000.0),
        },
        index=dates,
    )
    raw.index.name = "date"

    positions = np.arange(size, dtype=np.float64)
    snapshot = pd.DataFrame(
        {
            "feature_trend": np.sin(positions / 7.0),
            "feature_cycle": np.cos(positions / 11.0),
            "feature_phase": positions % 5.0,
        },
        index=dates[:-1],
    )
    snapshot.index.name = "date"
    snapshot["target_next_return"] = (
        raw["close"].shift(-1).iloc[:-1].to_numpy() / raw["close"].iloc[:-1].to_numpy() - 1.0
    )
    snapshot["target_next_direction"] = np.where(
        snapshot["target_next_return"] > 0.0,
        "UP",
        "DOWN",
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    raw_path = source_dir / "raw_ohlcv.csv"
    snapshot_path = source_dir / "dataset_snapshot.csv"
    raw.to_csv(raw_path, index_label="date", float_format="%.17g")
    snapshot.to_csv(snapshot_path, index_label="date", float_format="%.17g")

    source_manifest = {
        "schema_version": "1.0",
        "experiment": {"ticker": "TEST", "random_seed": 28},
        "artifacts": {
            "raw_ohlcv": {
                "file": raw_path.name,
                "sha256": digest(raw_path),
                "bytes": raw_path.stat().st_size,
            },
            "dataset_snapshot": {
                "file": snapshot_path.name,
                "sha256": digest(snapshot_path),
                "bytes": snapshot_path.stat().st_size,
            },
        },
        "dataset": {
            "feature_names": ["feature_trend", "feature_cycle", "feature_phase"],
            "usable_feature_rows": size,
        },
    }
    (source_dir / "dataset_manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )
    result = build_execution_dataset(
        raw_path,
        snapshot_path,
        output_root=tmp_path / "d04",
        threshold_bps=10.0,
    )
    return Path(result["output_directory"]) / "execution_dataset.csv"


def test_neutral_threshold_policy_changes_decisions_not_probability_inputs():
    probabilities = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.50, 0.30],
            [0.10, 0.20, 0.70],
        ]
    )
    assert apply_neutral_threshold(probabilities, 0.40) == (
        "DOWN",
        "NEUTRAL",
        "UP",
    )
    assert apply_neutral_threshold(probabilities, 0.60) == (
        "DOWN",
        "UP",
        "UP",
    )


def test_threshold_selection_is_deterministic_and_uses_three_classes():
    actual = np.asarray(["DOWN", "DOWN", "NEUTRAL", "NEUTRAL", "UP", "UP"])
    probabilities = np.asarray(
        [
            [0.70, 0.10, 0.20],
            [0.65, 0.15, 0.20],
            [0.25, 0.45, 0.30],
            [0.20, 0.55, 0.25],
            [0.20, 0.15, 0.65],
            [0.15, 0.20, 0.65],
        ]
    )
    first = choose_neutral_threshold(actual, probabilities)
    second = choose_neutral_threshold(actual, probabilities)
    assert first == second
    assert 0.05 <= first["threshold"] <= 0.80
    assert 0.0 <= first["macro_f1"] <= 1.0
    assert 0.0 <= first["balanced_accuracy"] <= 1.0
    assert 0.0 <= first["neutral_precision"] <= 1.0
    assert 0.0 <= first["neutral_recall"] <= 1.0


def test_threshold_selection_rejects_missing_class_and_invalid_probabilities():
    actual = np.asarray(["DOWN", "DOWN", "UP", "UP"])
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
        ]
    )
    with pytest.raises(ValueError, match="all three classes"):
        choose_neutral_threshold(actual, probabilities)
    with pytest.raises(ValueError, match="sum to one"):
        choose_neutral_threshold(
            np.asarray(["DOWN", "NEUTRAL", "UP"]),
            np.asarray([[0.6, 0.6, 0.0], [0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]),
        )
    with pytest.raises(ValueError, match="finite"):
        choose_neutral_threshold(
            np.asarray(["DOWN", "NEUTRAL", "UP"]),
            np.asarray([[0.5, 0.5, 0.0], [0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]),
            thresholds=(0.2, float("nan")),
        )


def test_nested_policy_keeps_outer_test_out_of_threshold_selection():
    dataset = classification_dataset()
    fold = build_expanding_folds(120)[0]
    dates = tuple(f"row-{index}" for index in range(120))

    record, predictions = evaluate_decision_policy_fold(
        dataset,
        fold,
        algorithm="logistic_regression",
        seed=28,
        dates=dates,
        calibration_fraction=0.20,
    )
    assert record["status"] == "COMPLETED"
    assert record["test_used_for_selection"] is False
    assert record["training_samples"] == 72
    assert record["test_samples"] == 12
    assert record["train_end_index"] == 72
    assert record["test_start_index"] == 72
    assert record["test_end_index"] == 84
    assert record["calibration_start_index"] == 57
    assert record["calibration_end_index"] == 72
    assert record["calibration_samples"] == 15
    assert record["calibration_test_used_for_selection"] is True
    assert record["probability_calibration_method"] == "multiclass_temperature_scaling"
    assert 0.05 <= record["probability_calibration_temperature"] <= 20.0
    assert record["calibration_scaled_log_loss"] <= record["calibration_raw_log_loss"] + 1e-8
    assert 0.05 <= record["selected_neutral_threshold"] <= 0.80
    assert len(predictions) == 12
    assert predictions[0]["row_index"] == 72
    assert predictions[-1]["row_index"] == 83
    for row in predictions:
        assert row["feature_date"] in dates[72:84]
        assert row["policy_prediction"] in {"DOWN", "NEUTRAL", "UP"}
        assert row["probability_down"] + row["probability_neutral"] + row[
            "probability_up"
        ] == pytest.approx(1.0)
    assert "policy_neutral_recall" in record
    assert "argmax_neutral_recall" in record
    assert "multiclass_log_loss" in record


def test_outer_test_changes_cannot_change_selected_calibration_threshold():
    dataset = classification_dataset()
    fold = build_expanding_folds(120)[0]
    dates = tuple(f"row-{index}" for index in range(120))
    original, _ = evaluate_decision_policy_fold(
        dataset,
        fold,
        algorithm="logistic_regression",
        seed=28,
        dates=dates,
    )

    changed_target = list(dataset.target)
    for index in range(fold.test_start, fold.test_end):
        label = changed_target[index]
        changed_target[index] = {
            "DOWN": "NEUTRAL",
            "NEUTRAL": "UP",
            "UP": "DOWN",
        }[label]
    changed_features = list(dataset.features)
    for index in range(fold.test_start, fold.test_end):
        changed_features[index] = (1_000_000.0, -1_000_000.0, 0.0)
    altered = MLDataset(
        dataset_id=dataset.dataset_id,
        source=dataset.source,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        features=tuple(changed_features),
        target=tuple(changed_target),
        task="classification",
    )
    modified, _ = evaluate_decision_policy_fold(
        altered,
        fold,
        algorithm="logistic_regression",
        seed=28,
        dates=dates,
    )
    assert modified["selected_neutral_threshold"] == original["selected_neutral_threshold"]
    assert (
        modified["probability_calibration_temperature"]
        == original["probability_calibration_temperature"]
    )
    assert modified["calibration_macro_f1"] == original["calibration_macro_f1"]
    assert modified["calibration_start_index"] == original["calibration_start_index"]


def test_policy_rejects_invalid_probabilities_and_thresholds():
    with pytest.raises(ValueError, match="sum to one"):
        apply_neutral_threshold(np.asarray([[0.4, 0.4, 0.4]]), 0.3)
    with pytest.raises(ValueError, match="between zero and one"):
        apply_neutral_threshold(np.asarray([[0.3, 0.4, 0.3]]), 1.5)
    with pytest.raises(ValueError, match="finite"):
        apply_neutral_threshold(np.asarray([[0.3, float("nan"), 0.7]]), 0.3)


def test_run_writes_complete_hash_manifest_compatible_with_d07(tmp_path, monkeypatch):
    dataset_path = create_d04_dataset(tmp_path)
    monkeypatch.setattr(
        "scripts.run_execution_decision_policy.DEFAULT_ALGORITHMS",
        ("logistic_regression",),
    )
    summary = run_decision_policy(
        dataset_path,
        output_root=tmp_path / "d06",
        seed=28,
        n_folds=1,
        test_fraction=0.10,
        calibration_fraction=0.20,
    )
    run_dir = Path(summary["output_directory"])
    manifest_path = run_dir / "decision_policy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    d04_manifest_path = dataset_path.parent / "execution_dataset_manifest.json"

    assert summary["status"] == "COMPLETED"
    assert summary["completed_model_folds"] == 1
    assert summary["failed_model_folds"] == 0
    assert summary["prediction_rows"] == 18
    assert manifest["contract_version"] == "D06.3"
    assert manifest["configuration"]["probability_calibration_method"] == (
        "multiclass_temperature_scaling"
    )
    assert manifest["configuration"]["reliability_bin_count"] == 10
    assert len(summary["reliability_bins"]) == 30
    assert summary["probability_calibration_method"] == "multiclass_temperature_scaling"
    assert manifest["input"]["execution_dataset"] == dataset_path.name
    assert manifest["input"]["execution_dataset_sha256"] == digest(dataset_path)
    assert manifest["input"]["source_manifest_sha256"] == digest(d04_manifest_path)
    assert manifest["configuration"]["algorithms"] == ["logistic_regression"]
    for name, details in manifest["outputs"].items():
        path = run_dir / name
        assert path.is_file()
        assert details["sha256"] == digest(path)
        assert details["bytes"] == path.stat().st_size

    predictions = pd.read_csv(run_dir / "oos_predictions.csv")
    assert len(predictions) == 18
    assert set(predictions["actual"]) == {"DOWN", "NEUTRAL", "UP"}
    assert set(predictions["argmax_prediction"]).issubset({"DOWN", "NEUTRAL", "UP"})
    assert set(predictions["policy_prediction"]).issubset({"DOWN", "NEUTRAL", "UP"})
    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    assert {
        "multiclass_expected_calibration_error",
        "multiclass_maximum_calibration_error",
    }.issubset(fold_metrics.columns)



def test_temperature_scaling_preserves_probability_simplex_and_sharpens_confidence():
    raw = np.asarray(
        [
            [0.55, 0.30, 0.15],
            [0.20, 0.50, 0.30],
        ],
        dtype=np.float64,
    )

    softened = apply_temperature_scaling(raw, 2.0)
    sharpened = apply_temperature_scaling(raw, 0.5)

    assert softened.sum(axis=1) == pytest.approx(np.ones(2))
    assert sharpened.sum(axis=1) == pytest.approx(np.ones(2))
    assert sharpened[0, 0] > raw[0, 0]
    assert softened[0, 0] < raw[0, 0]


def test_temperature_fit_does_not_increase_calibration_log_loss():
    actual = np.asarray(["DOWN", "NEUTRAL", "UP"] * 12)
    probabilities = np.tile(
        np.asarray(
            [
                [0.55, 0.25, 0.20],
                [0.20, 0.55, 0.25],
                [0.20, 0.25, 0.55],
            ]
        ),
        (12, 1),
    )

    temperature, diagnostics = fit_temperature_scaling(actual, probabilities)

    assert 0.05 <= temperature <= 20.0
    assert diagnostics["calibrated_log_loss"] <= diagnostics["raw_log_loss"] + 1e-8
    calibrated = apply_temperature_scaling(probabilities, temperature)
    assert calibrated.sum(axis=1) == pytest.approx(np.ones(len(actual)))


def test_temperature_fit_rejects_calibration_without_all_classes():
    actual = np.asarray(["DOWN", "DOWN", "UP", "UP"])
    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.3, 0.5],
        ]
    )

    with pytest.raises(ValueError, match="all three classes"):
        fit_temperature_scaling(actual, probabilities)
