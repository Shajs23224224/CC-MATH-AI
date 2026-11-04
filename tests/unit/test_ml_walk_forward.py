"""Tests for ML expanding-window folds and out-of-sample predictions."""

import pytest

from core.contracts.ml import MLDataset
from scripts.run_ml_walk_forward import (
    build_expanding_folds,
    evaluate_model_fold,
)


def regression_dataset(size: int = 100) -> MLDataset:
    features = tuple(
        (float(index), float(index % 7))
        for index in range(size)
    )
    target = tuple(
        0.015 * index + 0.2 * (index % 7)
        for index in range(size)
    )

    return MLDataset(
        dataset_id="walk-forward-regression-test",
        source="synthetic temporal test fixture",
        feature_names=("trend", "cycle"),
        target_name="return",
        features=features,
        target=target,
        task="regression",
    )


def classification_dataset(size: int = 100) -> MLDataset:
    features = tuple(
        (float(index), float(index % 2))
        for index in range(size)
    )
    target = tuple(
        "UP" if index % 2 else "DOWN"
        for index in range(size)
    )

    return MLDataset(
        dataset_id="walk-forward-classification-test",
        source="synthetic temporal test fixture",
        feature_names=("trend", "parity"),
        target_name="direction",
        features=features,
        target=target,
        task="classification",
    )


def test_expanding_folds_are_contiguous_and_non_overlapping() -> None:
    folds = build_expanding_folds(2490)

    assert len(folds) == 4
    assert [
        (fold.train_end, fold.test_start, fold.test_end)
        for fold in folds
    ] == [
        (1494, 1494, 1743),
        (1743, 1743, 1992),
        (1992, 1992, 2241),
        (2241, 2241, 2490),
    ]

    assert all(fold.train_start == 0 for fold in folds)
    assert all(fold.test_samples == 249 for fold in folds)


def test_expanding_folds_reject_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="test window"):
        build_expanding_folds(20, n_folds=4, test_fraction=0.01)

    with pytest.raises(ValueError, match="not enough samples"):
        build_expanding_folds(40, n_folds=4, test_fraction=0.4)


def test_regression_predictions_respect_fold_boundaries() -> None:
    dataset = regression_dataset()
    fold = build_expanding_folds(100)[0]
    dates = tuple(f"row-{index}" for index in range(100))

    record, predictions = evaluate_model_fold(
        dataset,
        fold,
        algorithm="ridge",
        seed=28,
        dates=dates,
    )

    assert record["status"] == "COMPLETED"
    assert record["training_samples"] == 60
    assert record["test_samples"] == 10
    assert record["train_end_index"] == 60
    assert record["test_start_index"] == 60
    assert record["test_end_index"] == 70
    assert record["test_used_for_selection"] is False

    assert len(predictions) == 10
    assert predictions[0]["row_index"] == 60
    assert predictions[-1]["row_index"] == 69
    assert predictions[0]["feature_date"] == dates[60]


def test_classification_predictions_include_probabilities() -> None:
    dataset = classification_dataset()
    fold = build_expanding_folds(100)[0]
    dates = tuple(f"row-{index}" for index in range(100))

    record, predictions = evaluate_model_fold(
        dataset,
        fold,
        algorithm="logistic_regression",
        seed=28,
        dates=dates,
    )

    assert record["status"] == "COMPLETED"
    assert len(predictions) == 10

    for prediction in predictions:
        assert "probability_down" in prediction
        assert "probability_up" in prediction
        assert (
            prediction["probability_down"] + prediction["probability_up"]
            == pytest.approx(1.0)
        )
