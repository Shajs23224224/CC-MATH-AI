"""Behavioral tests for temporal gradient boosting integration."""

import pytest

from core.contracts.ml import MLDataset, MLTrainingConfig
from ml import ClassicalMLModel, supported_algorithms
from ml.errors import MLInputError


def regression_dataset() -> MLDataset:
    features = tuple((float(index), float(index % 7)) for index in range(120))
    target = tuple(0.02 * index + (index % 7) * 0.15 for index in range(120))
    return MLDataset(
        dataset_id="f29-synthetic-regression-v1",
        source="synthetic F29 unit test",
        feature_names=("trend", "cycle"),
        target_name="target",
        features=features,
        target=target,
        task="regression",
    )


def classification_dataset() -> MLDataset:
    features = tuple((float(index), float(index % 7)) for index in range(120))
    target = tuple("positive" if index % 2 else "negative" for index in range(120))
    return MLDataset(
        dataset_id="f29-synthetic-classification-v1",
        source="synthetic F29 unit test",
        feature_names=("trend", "cycle"),
        target_name="label",
        features=features,
        target=target,
        task="classification",
    )


def test_gradient_boosting_algorithms_are_registered_by_task() -> None:
    registry = supported_algorithms()
    assert "gradient_boosting_regressor" in registry["regression"]
    assert "gradient_boosting_classifier" in registry["classification"]


def test_regressor_selects_tree_count_on_temporal_validation_and_predicts() -> None:
    model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm="gradient_boosting_regressor",
            task="regression",
            random_seed=29,
            test_fraction=0.2,
            parameters={"n_estimators": 40, "max_depth": 2},
        )
    )
    result = model.fit(regression_dataset())

    assert result.split_method == "chronological"
    assert result.train_end_index == result.test_start_index == 96
    assert result.test_end_index == 120
    assert result.test_used_for_selection is False
    assert set(result.metrics) == {"mae", "rmse", "r2"}
    assert model.metadata is not None
    assert model.metadata.parameters["temporal_early_stopping"] is True
    assert model.metadata.parameters["requested_max_estimators"] == 40
    selected = model.metadata.parameters["selected_n_estimators"]
    assert isinstance(selected, int)
    assert 1 <= selected <= 40
    assert model.metadata.parameters["early_stopping_validation_samples"] == 15

    prediction = model.predict(((121.0, 2.0), (122.0, 3.0)))
    assert len(prediction.predictions) == 2
    assert all(isinstance(value, float) for value in prediction.predictions)


def test_classifier_returns_normalized_probabilities_and_audits_stopping() -> None:
    model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm="gradient_boosting_classifier",
            task="classification",
            random_seed=29,
            parameters={"n_estimators": 30, "max_depth": 2},
        )
    )
    result = model.fit(classification_dataset())
    prediction = model.predict(((121.0, 2.0), (122.0, 3.0)))

    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert prediction.classes is not None
    assert prediction.probabilities is not None
    assert len(prediction.probabilities) == len(prediction.predictions) == 2
    assert all(abs(sum(row) - 1.0) < 1e-8 for row in prediction.probabilities)
    assert model.metadata is not None
    assert model.metadata.parameters["temporal_early_stopping"] is True
    assert model.metadata.parameters["requested_max_estimators"] == 30


def test_test_tail_targets_cannot_change_selected_tree_count() -> None:
    original = regression_dataset()
    altered_target = list(original.target)
    altered_target[96:] = tuple(100_000.0 + index for index in range(24))
    altered = original.model_copy(update={"target": tuple(altered_target)})
    config = MLTrainingConfig(
        algorithm="gradient_boosting_regressor",
        task="regression",
        random_seed=7,
        parameters={"n_estimators": 35},
    )

    original_model = ClassicalMLModel(config)
    altered_model = ClassicalMLModel(config)
    original_model.fit(original)
    altered_model.fit(altered)

    assert original_model.metadata is not None
    assert altered_model.metadata is not None
    assert (
        original_model.metadata.parameters["selected_n_estimators"]
        == altered_model.metadata.parameters["selected_n_estimators"]
    )
    assert original_model.metadata.data_fingerprint == altered_model.metadata.data_fingerprint


def test_gradient_boosting_rejects_unbounded_tree_budget() -> None:
    model = ClassicalMLModel(
        MLTrainingConfig(
            algorithm="gradient_boosting_regressor",
            task="regression",
            parameters={"n_estimators": 1_001},
        )
    )
    with pytest.raises(MLInputError, match="between 1 and 1000"):
        model.fit(regression_dataset())
