"""Behavioral tests for leakage-aware F30 unsupervised learning."""

import math

import pytest

from core.contracts.unsupervised import UnsupervisedMLConfig
from ml import UnsupervisedMLModel, supported_unsupervised_algorithms
from ml.errors import MLInputError


def sample_features() -> tuple[tuple[float, ...], ...]:
    rows = []
    for index in range(120):
        offset = 12.0 if index >= 60 else 0.0
        rows.append(
            (
                float(index % 10) + offset,
                float((index * 7) % 17) * 0.4 + offset * 0.7,
                math.sin(index / 3.0) + offset * 0.03,
            )
        )
    return tuple(rows)


def test_registry_is_explicit_and_operation_scoped() -> None:
    registry = supported_unsupervised_algorithms()
    assert set(registry) == {"clustering", "dimensionality_reduction", "anomaly_detection"}
    assert set(registry["clustering"]) == {
        "kmeans",
        "agglomerative_clustering",
        "dbscan",
        "gaussian_mixture",
    }
    assert set(registry["dimensionality_reduction"]) == {"pca", "kernel_pca", "fast_ica"}
    assert set(registry["anomaly_detection"]) == {
        "isolation_forest",
        "local_outlier_factor",
        "elliptic_envelope",
    }


@pytest.mark.parametrize(
    ("algorithm", "supports_holdout_labels"),
    [
        ("kmeans", True),
        ("agglomerative_clustering", False),
        ("dbscan", False),
        ("gaussian_mixture", True),
    ],
)
def test_clustering_outputs_training_labels_and_stability(
    algorithm: str, supports_holdout_labels: bool
) -> None:
    model = UnsupervisedMLModel(
        UnsupervisedMLConfig(
            algorithm=algorithm,
            random_seed=30,
            stability_repeats=2,
            parameters=(
                {"n_clusters": 2}
                if algorithm in {"kmeans", "agglomerative_clustering"}
                else {"n_components": 2}
                if algorithm == "gaussian_mixture"
                else {}
            ),
        )
    )
    result = model.fit(sample_features(), ("x", "y", "z"))
    assert result.operation == "clustering"
    assert len(result.training_labels or ()) == result.training_observations
    assert (result.holdout_labels is not None) is supports_holdout_labels
    assert len(result.stability_scores) == 2
    assert result.stability_mean_ari is not None
    assert all(-1.0 <= score <= 1.0 for score in result.stability_scores)
    assert result.holdout_used_for_fit is False
    assert any("not causal" in item for item in result.limitations)


@pytest.mark.parametrize("algorithm", ["pca", "kernel_pca", "fast_ica"])
def test_reduction_fits_training_prefix_and_transforms_holdout(algorithm: str) -> None:
    result = UnsupervisedMLModel(UnsupervisedMLConfig(algorithm=algorithm, random_seed=30)).fit(
        sample_features(), ("x", "y", "z")
    )
    assert result.operation == "dimensionality_reduction"
    assert len(result.training_embedding or ()) == result.training_observations
    assert len(result.holdout_embedding or ()) == result.holdout_observations
    assert result.training_embedding is not None
    assert len(result.training_embedding[0]) == 2
    assert result.scaler_fit_scope == "training_prefix_only"
    assert result.stability_scores == ()


@pytest.mark.parametrize(
    "algorithm", ["isolation_forest", "local_outlier_factor", "elliptic_envelope"]
)
def test_anomaly_detectors_return_finite_scores_and_binary_flags(algorithm: str) -> None:
    result = UnsupervisedMLModel(UnsupervisedMLConfig(algorithm=algorithm, random_seed=30)).fit(
        sample_features(), ("x", "y", "z")
    )
    assert result.operation == "anomaly_detection"
    assert len(result.training_anomaly_scores or ()) == result.training_observations
    assert len(result.holdout_anomaly_scores or ()) == result.holdout_observations
    assert all(math.isfinite(score) for score in result.training_anomaly_scores or ())
    assert set(result.training_anomaly_flags or ()) <= {-1, 1}
    assert set(result.holdout_anomaly_flags or ()) <= {-1, 1}


def test_holdout_changes_do_not_change_scaling_or_training_fingerprint() -> None:
    original = sample_features()
    altered = list(original)
    for index in range(96, len(altered)):
        altered[index] = (altered[index][0] + 10_000.0, altered[index][1] - 8_000.0, 500.0)
    config = UnsupervisedMLConfig(
        algorithm="kmeans",
        random_seed=30,
        stability_repeats=2,
        parameters={"n_clusters": 2},
    )
    first = UnsupervisedMLModel(config).fit(original, ("x", "y", "z"))
    second = UnsupervisedMLModel(config).fit(tuple(altered), ("x", "y", "z"))
    assert first.data_fingerprint == second.data_fingerprint
    assert first.scaler_mean == second.scaler_mean
    assert first.scaler_scale == second.scaler_scale
    assert first.training_labels == second.training_labels
    assert first.stability_scores == second.stability_scores


def test_rejects_bad_input_and_unsafe_resource_configuration() -> None:
    with pytest.raises(MLInputError, match="finite"):
        invalid = list(sample_features())
        invalid[10] = (float("nan"), 1.0, 2.0)
        UnsupervisedMLModel(UnsupervisedMLConfig(algorithm="pca")).fit(
            tuple(invalid), ("x", "y", "z")
        )
    with pytest.raises(ValueError, match="random_state"):
        UnsupervisedMLConfig(algorithm="kmeans", parameters={"random_state": 99})
    with pytest.raises(ValueError, match="n_estimators"):
        UnsupervisedMLConfig(algorithm="isolation_forest", parameters={"n_estimators": 5_000})
    with pytest.raises(MLInputError, match="feature_names"):
        UnsupervisedMLModel(UnsupervisedMLConfig(algorithm="pca")).fit(
            sample_features(), ("only_one_name",)
        )
