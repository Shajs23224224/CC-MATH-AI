"""Behavioral tests for F31 architectures, temporal boundaries and optional inference."""

from __future__ import annotations

from importlib.util import find_spec
from math import isfinite

import pytest

from core.contracts.deep_learning import DeepLearningConfig
from core.contracts.ml import MLDataset
from ml import DeepLearningModel, supported_deep_learning_algorithms
from ml.errors import MLInputError, MLPredictionError

pytestmark = pytest.mark.skipif(
    find_spec("torch") is None,
    reason="install the optional deep-learning extra",
)


def regression_dataset(n: int = 160) -> MLDataset:
    features = tuple(
        (float(index) / n, float(index % 7) / 7.0, float((index * 3) % 11) / 11.0)
        for index in range(n)
    )
    target = tuple(0.5 * row[0] + 0.2 * row[1] - 0.1 * row[2] for row in features)
    return MLDataset(
        dataset_id=f"f31-synthetic-regression-{n}",
        source="synthetic F31 unit fixture",
        feature_names=("trend", "cycle", "cross_cycle"),
        target_name="next_value",
        features=features,
        target=target,
        task="regression",
    )


def classification_dataset(n: int = 160) -> MLDataset:
    features = tuple(
        (float(index) / n, float(index % 7) / 7.0, float((index * 3) % 11) / 11.0)
        for index in range(n)
    )
    target = tuple("positive" if index % 2 else "negative" for index in range(n))
    return MLDataset(
        dataset_id=f"f31-synthetic-classification-{n}",
        source="synthetic F31 unit fixture",
        feature_names=("trend", "cycle", "cross_cycle"),
        target_name="direction",
        features=features,
        target=target,
        task="classification",
    )


def compact_config(algorithm: str, task: str) -> DeepLearningConfig:
    return DeepLearningConfig(
        algorithm=algorithm,
        task=task,
        random_seed=31,
        test_fraction=0.2,
        validation_fraction=0.15,
        sequence_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        learning_rate=0.003,
        batch_size=16,
        epochs=3,
        patience=2,
        max_parameters=200_000,
    )


def test_registry_is_explicit_and_task_scoped() -> None:
    registry = supported_deep_learning_algorithms()
    assert set(registry) == {"supervised", "reconstruction"}
    assert set(registry["supervised"]) == {
        "mlp",
        "lstm",
        "gru",
        "tcn",
        "attention",
        "transformer",
        "temporal_fusion_transformer",
    }
    assert set(registry["reconstruction"]) == {"autoencoder", "vae"}


@pytest.mark.parametrize(
    "algorithm",
    [
        "mlp",
        "lstm",
        "gru",
        "tcn",
        "attention",
        "transformer",
        "temporal_fusion_transformer",
    ],
)
def test_supervised_architecture_fits_temporal_splits_and_predicts(algorithm: str) -> None:
    dataset = regression_dataset()
    model = DeepLearningModel(compact_config(algorithm, "forecasting"))
    result = model.fit(dataset)

    assert result.algorithm == algorithm
    assert result.task == "forecasting"
    assert result.train_end_index == result.validation_start_index
    assert result.validation_end_index == result.test_start_index
    assert result.test_end_index == result.total_observations
    assert result.validation_used_for_early_stopping is True
    assert result.test_used_for_selection is False
    assert result.scaler_fit_scope == "training_features_only"
    assert result.scaler_fit_end_index < result.train_end_index
    assert result.training_samples == result.train_end_index - result.sequence_length
    assert len(result.test_predictions) == result.test_samples
    assert len(result.test_targets) == result.test_samples
    assert result.parameter_count <= model.config.max_parameters
    assert result.epochs_completed <= model.config.epochs
    assert all(isfinite(value) for value in result.metrics.values())
    prediction = model.predict(dataset.features[-8:])
    assert len(prediction.predictions) == 8 - model.config.sequence_length + 1
    assert all(isinstance(value, float) for value in prediction.predictions)


def test_classification_returns_classes_and_normalized_probabilities() -> None:
    dataset = classification_dataset()
    model = DeepLearningModel(compact_config("mlp", "classification"))
    result = model.fit(dataset)
    prediction = model.predict(dataset.features[-8:])

    assert result.metrics["accuracy"] >= 0.0
    assert 0.0 <= result.metrics["f1_macro"] <= 1.0
    assert prediction.classes is not None
    assert prediction.probabilities is not None
    assert len(prediction.probabilities) == len(prediction.predictions)
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in prediction.probabilities)
    assert set(prediction.predictions) <= {"positive", "negative"}


@pytest.mark.parametrize("algorithm", ["autoencoder", "vae"])
def test_reconstruction_architectures_return_finite_test_error_and_windows(
    algorithm: str,
) -> None:
    dataset = regression_dataset()
    model = DeepLearningModel(compact_config(algorithm, "reconstruction"))
    result = model.fit(dataset)

    assert result.task == "reconstruction"
    assert result.test_predictions == ()
    assert result.test_targets == ()
    assert isfinite(result.metrics["reconstruction_mse"])
    reconstructed = model.reconstruct(dataset.features[-8:])
    assert len(reconstructed) == 8 - model.config.sequence_length + 1
    assert len(reconstructed[0]) == model.config.sequence_length * len(dataset.feature_names)
    assert all(isfinite(value) for row in reconstructed for value in row)
    with pytest.raises(MLPredictionError, match="reconstruct"):
        model.predict(dataset.features[-8:])


def test_test_target_changes_do_not_change_training_fingerprint_or_selected_epoch() -> None:
    original = regression_dataset()
    altered_target = list(original.target)
    altered_target[-32:] = tuple(10_000.0 + index for index in range(32))
    altered = original.model_copy(update={"target": tuple(altered_target)})
    config = compact_config("mlp", "forecasting")

    first = DeepLearningModel(config)
    second = DeepLearningModel(config)
    first_result = first.fit(original)
    second_result = second.fit(altered)

    assert first_result.data_fingerprint == second_result.data_fingerprint
    assert first_result.best_epoch == second_result.best_epoch
    assert first_result.best_validation_loss == second_result.best_validation_loss
    assert first_result.test_predictions == second_result.test_predictions
    assert first_result.metrics != second_result.metrics


def test_configuration_rejects_incompatible_architecture_and_head_dimensions() -> None:
    with pytest.raises(ValueError, match="autoencoder and vae require reconstruction"):
        DeepLearningConfig(algorithm="vae", task="forecasting")
    with pytest.raises(ValueError, match="divisible by num_heads"):
        DeepLearningConfig(
            algorithm="transformer",
            task="forecasting",
            hidden_dim=10,
            num_heads=3,
        )


def test_resource_budget_and_non_finite_prediction_inputs_are_rejected() -> None:
    dataset = regression_dataset()
    bounded = DeepLearningConfig(
        algorithm="mlp",
        task="forecasting",
        sequence_length=8,
        hidden_dim=256,
        num_layers=4,
        max_parameters=1_000,
    )
    model = DeepLearningModel(bounded)
    with pytest.raises(MLInputError, match="parameters"):
        model.fit(dataset)

    fitted = DeepLearningModel(compact_config("mlp", "forecasting"))
    fitted.fit(dataset)
    invalid = list(dataset.features[-8:])
    invalid[-1] = (float("nan"), invalid[-1][1], invalid[-1][2])
    with pytest.raises(MLInputError, match="finite"):
        fitted.predict(tuple(invalid))
