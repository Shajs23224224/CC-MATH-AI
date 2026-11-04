"""Tests for nested temporal calibration of three-class decision policies."""

import numpy as np
import pytest

from core.contracts.ml import MLDataset
from scripts.run_execution_decision_policy import (
    apply_neutral_threshold,
    choose_neutral_threshold,
    evaluate_decision_policy_fold,
)
from scripts.run_ml_walk_forward import build_expanding_folds


def classification_dataset(size=120):
    features = tuple(
        (float(index), float(index % 3), float((index % 7) / 7.0))
        for index in range(size)
    )
    labels = tuple(
        ("DOWN", "NEUTRAL", "UP")[index % 3]
        for index in range(size)
    )

    return MLDataset(
        dataset_id="decision-policy-synthetic",
        source="synthetic chronological test fixture",
        feature_names=("trend", "cycle", "range"),
        target_name="direction",
        features=features,
        target=labels,
        task="classification",
    )


def test_neutral_threshold_changes_only_decision_policy():
    probabilities = np.asarray([
        [0.70, 0.20, 0.10],
        [0.20, 0.50, 0.30],
        [0.10, 0.20, 0.70],
    ])

    assert apply_neutral_threshold(
        probabilities,
        0.40,
    ) == ("DOWN", "NEUTRAL", "UP")

    assert apply_neutral_threshold(
        probabilities,
        0.60,
    ) == ("DOWN", "UP", "UP")


def test_threshold_selection_uses_valid_three_class_inputs():
    actual = np.asarray([
        "DOWN", "DOWN", "NEUTRAL", "NEUTRAL", "UP", "UP"
    ])

    probabilities = np.asarray([
        [0.70, 0.10, 0.20],
        [0.65, 0.15, 0.20],
        [0.25, 0.45, 0.30],
        [0.20, 0.55, 0.25],
        [0.20, 0.15, 0.65],
        [0.15, 0.20, 0.65],
    ])

    result = choose_neutral_threshold(actual, probabilities)

    assert 0.05 <= result["threshold"] <= 0.80
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert 0.0 <= result["balanced_accuracy"] <= 1.0
    assert 0.0 <= result["neutral_precision"] <= 1.0
    assert 0.0 <= result["neutral_recall"] <= 1.0


def test_threshold_selection_rejects_missing_calibration_class():
    actual = np.asarray(["DOWN", "DOWN", "UP", "UP"])
    probabilities = np.asarray([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.1, 0.7],
    ])

    with pytest.raises(ValueError, match="all three classes"):
        choose_neutral_threshold(actual, probabilities)


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

    assert 0.05 <= record["selected_neutral_threshold"] <= 0.80
    assert len(predictions) == 12
    assert predictions[0]["row_index"] == 72
    assert predictions[-1]["row_index"] == 83

    for row in predictions:
        assert row["feature_date"] in dates[72:84]
        assert row["policy_prediction"] in {"DOWN", "NEUTRAL", "UP"}
        assert (
            row["probability_down"]
            + row["probability_neutral"]
            + row["probability_up"]
            == pytest.approx(1.0)
        )

    assert "policy_neutral_recall" in record
    assert "argmax_neutral_recall" in record
    assert "multiclass_log_loss" in record


def test_policy_rejects_invalid_probabilities_and_thresholds():
    with pytest.raises(ValueError, match="sum to one"):
        apply_neutral_threshold(
            np.asarray([[0.4, 0.4, 0.4]]),
            0.3,
        )

    with pytest.raises(ValueError, match="between zero and one"):
        apply_neutral_threshold(
            np.asarray([[0.3, 0.4, 0.3]]),
            1.5,
        )
