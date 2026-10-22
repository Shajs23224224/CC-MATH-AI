"""Behavioral tests for classical model training, temporal splits and predictions."""

import numpy as np
import pytest

from core.contracts.ml import MLDataset, MLTrainingConfig
from ml import ClassicalMLModel, supported_algorithms
from ml.errors import MLInputError


def regression_dataset() -> MLDataset:
    x = tuple((float(value),) for value in range(60))
    y = tuple(3.0 * value - 2.0 for value in range(60))
    return MLDataset(
        dataset_id="synthetic-linear-v1",
        source="synthetic test fixture",
        feature_names=("x",),
        target_name="y",
        features=x,
        target=y,
        task="regression",
    )


def test_supported_models_are_task_scoped() -> None:
    registry = supported_algorithms()
    assert "ridge" in registry["regression"]
    assert "random_forest_classifier" in registry["classification"]


def test_ridge_fits_and_reports_chronological_oos_metrics() -> None:
    model = ClassicalMLModel(MLTrainingConfig(algorithm="ridge", task="regression", random_seed=11))
    result = model.fit(regression_dataset())
    assert result.train_end_index == result.test_start_index
    assert result.test_end_index == 60
    assert result.test_used_for_selection is False
    assert result.metrics["rmse"] < 5.0
    assert model.metadata is not None
    assert model.metadata.training_observations == result.training_samples
    prediction = model.predict(((61.0,), (62.0,)))
    assert len(prediction.predictions) == 2
    assert all(np.isfinite(float(value)) for value in prediction.predictions)


def test_fit_is_reproducible_for_same_seed_and_data() -> None:
    config = MLTrainingConfig(
        algorithm="random_forest_regressor", task="regression", random_seed=28
    )
    first = ClassicalMLModel(config)
    second = ClassicalMLModel(config)
    result_a = first.fit(regression_dataset())
    result_b = second.fit(regression_dataset())
    assert result_a.metrics == result_b.metrics
    assert first.metadata is not None and second.metadata is not None
    assert first.metadata.data_fingerprint == second.metadata.data_fingerprint


def test_classifier_returns_probabilities_and_metrics() -> None:
    features = tuple((float(index), float(index % 3)) for index in range(60))
    labels = tuple("positive" if index % 2 else "negative" for index in range(60))
    dataset = MLDataset(
        dataset_id="synthetic-classification-v1",
        source="synthetic test fixture",
        feature_names=("trend", "cycle"),
        target_name="label",
        features=features,
        target=labels,
        task="classification",
    )
    model = ClassicalMLModel(
        MLTrainingConfig(algorithm="logistic_regression", task="classification", random_seed=5)
    )
    result = model.fit(dataset)
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    prediction = model.predict(((61.0, 1.0), (62.0, 2.0)))
    assert prediction.probabilities is not None
    assert prediction.classes is not None
    assert len(prediction.probabilities) == 2
    assert all(abs(sum(row) - 1.0) < 1e-8 for row in prediction.probabilities)


def test_invalid_parameters_and_unfitted_prediction_fail_closed() -> None:
    with pytest.raises(ValueError, match="random_state"):
        MLTrainingConfig(algorithm="ridge", task="regression", parameters={"random_state": 4})
    model = ClassicalMLModel(MLTrainingConfig(algorithm="ridge", task="regression"))
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(((1.0,),))
    with pytest.raises(MLInputError, match="unsupported"):
        ClassicalMLModel(MLTrainingConfig(algorithm="unknown", task="regression")).fit(
            regression_dataset()
        )


def test_rejects_misaligned_or_non_finite_dataset() -> None:
    with pytest.raises(ValueError, match="same number"):
        MLDataset(
            dataset_id="invalid",
            source="unit test",
            feature_names=("x",),
            target_name="y",
            features=((1.0,),) * 13,
            target=(1.0,) * 12,
            task="regression",
        )
    with pytest.raises(ValueError):
        MLDataset(
            dataset_id="non-finite",
            source="unit test",
            feature_names=("x",),
            target_name="y",
            features=((float("nan"),),) * 12,
            target=(1.0,) * 12,
            task="regression",
        )
