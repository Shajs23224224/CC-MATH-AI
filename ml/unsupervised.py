"""Reproducible, leakage-aware clustering, reduction and anomaly detection for F30."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from core.contracts.unsupervised import (
    UnsupervisedAlgorithm,
    UnsupervisedMLConfig,
    UnsupervisedMLResult,
    UnsupervisedOperation,
)
from .errors import MLInputError, MLTrainingError

EstimatorFactory = Callable[[int], Any]

_OPERATION_BY_ALGORITHM: dict[UnsupervisedAlgorithm, UnsupervisedOperation] = {
    "kmeans": "clustering",
    "agglomerative_clustering": "clustering",
    "dbscan": "clustering",
    "gaussian_mixture": "clustering",
    "pca": "dimensionality_reduction",
    "kernel_pca": "dimensionality_reduction",
    "fast_ica": "dimensionality_reduction",
    "isolation_forest": "anomaly_detection",
    "local_outlier_factor": "anomaly_detection",
    "elliptic_envelope": "anomaly_detection",
}

_FACTORIES: dict[UnsupervisedAlgorithm, EstimatorFactory] = {
    "kmeans": lambda seed: KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=seed),
    "agglomerative_clustering": lambda seed: AgglomerativeClustering(
        n_clusters=3, linkage="ward"
    ),
    "dbscan": lambda seed: DBSCAN(eps=0.5, min_samples=5, n_jobs=1),
    "gaussian_mixture": lambda seed: GaussianMixture(
        n_components=3, n_init=1, max_iter=100, random_state=seed
    ),
    "pca": lambda seed: PCA(n_components=2, random_state=seed),
    "kernel_pca": lambda seed: KernelPCA(
        n_components=2, kernel="rbf", eigen_solver="arpack", random_state=seed
    ),
    "fast_ica": lambda seed: FastICA(
        n_components=2, max_iter=500, tol=0.001, whiten="unit-variance", random_state=seed
    ),
    "isolation_forest": lambda seed: IsolationForest(
        contamination=0.05, n_estimators=100, n_jobs=1, random_state=seed
    ),
    "local_outlier_factor": lambda seed: LocalOutlierFactor(
        n_neighbors=20, novelty=True, contamination=0.05, n_jobs=1
    ),
    "elliptic_envelope": lambda seed: EllipticEnvelope(
        contamination=0.05, random_state=seed
    ),
}

_CLUSTERING_ALGORITHMS: set[UnsupervisedAlgorithm] = {
    "kmeans",
    "agglomerative_clustering",
    "dbscan",
    "gaussian_mixture",
}
_STABILITY_ALGORITHMS = _CLUSTERING_ALGORITHMS


def supported_unsupervised_algorithms() -> dict[str, tuple[str, ...]]:
    """Return the explicit allow-list grouped by operation."""
    grouped: dict[str, list[str]] = {
        "clustering": [],
        "dimensionality_reduction": [],
        "anomaly_detection": [],
    }
    for algorithm, operation in _OPERATION_BY_ALGORITHM.items():
        grouped[operation].append(algorithm)
    return {operation: tuple(sorted(algorithms)) for operation, algorithms in grouped.items()}


def _as_matrix(rows: Sequence[Sequence[float]], *, max_observations: int) -> np.ndarray:
    """Convert and validate a finite, rectangular numeric matrix."""
    try:
        matrix = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MLInputError("features must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 12 or matrix.shape[1] < 1:
        raise MLInputError("features must be a 2-D matrix with at least 12 rows and one column")
    if matrix.shape[0] > max_observations:
        raise MLInputError(f"observations exceed the configured maximum of {max_observations}")
    if matrix.shape[1] > 64:
        raise MLInputError("at most 64 features are supported")
    if not np.all(np.isfinite(matrix)):
        raise MLInputError("features must contain only finite values")
    return matrix


def _fingerprint(training_features: np.ndarray, feature_names: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"features": training_features.tolist(), "feature_names": list(feature_names)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primitive_parameters(estimator: Any) -> dict[str, int | float | str | bool | None]:
    """Keep the audit record portable and free of estimator objects."""
    values: dict[str, int | float | str | bool | None] = {}
    for key, value in estimator.get_params(deep=False).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            values[key] = value
    return values


class UnsupervisedMLModel:
    """Fit one allow-listed unsupervised estimator on a chronological training prefix."""

    def __init__(self, config: UnsupervisedMLConfig) -> None:
        self.config = config
        self.result: UnsupervisedMLResult | None = None
        self._estimator: Any | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: tuple[str, ...] | None = None

    def _build_estimator(self, seed: int, training_observations: int) -> Any:
        estimator = _FACTORIES[self.config.algorithm](seed)
        supplied = dict(self.config.parameters)
        unsupported = set(supplied) - set(estimator.get_params(deep=False))
        if unsupported:
            raise MLInputError(f"unsupported model parameters: {', '.join(sorted(unsupported))}")
        try:
            estimator.set_params(**supplied)
            resolved = estimator.get_params(deep=False)
            if "random_state" in resolved:
                estimator.set_params(random_state=seed)
            if "n_jobs" in resolved:
                estimator.set_params(n_jobs=1)
            if self.config.algorithm == "local_outlier_factor":
                neighbors = int(estimator.get_params(deep=False)["n_neighbors"])
                if neighbors >= training_observations:
                    estimator.set_params(n_neighbors=max(2, training_observations - 1))
            return estimator
        except (ValueError, TypeError) as exc:
            raise MLInputError("invalid unsupervised estimator parameters") from exc

    @staticmethod
    def _cluster_labels(estimator: Any, features: np.ndarray) -> np.ndarray:
        if hasattr(estimator, "predict"):
            labels = np.asarray(estimator.predict(features), dtype=np.int64)
        else:
            labels = np.asarray(estimator.labels_, dtype=np.int64)
        if labels.ndim != 1 or len(labels) != len(features):
            raise MLTrainingError("clustering estimator returned misaligned labels")
        return labels

    def _stability(
        self,
        estimator: Any,
        training_features: np.ndarray,
        training_labels: np.ndarray,
    ) -> tuple[float, ...]:
        """Compare resampled cluster assignments with full-prefix labels via adjusted Rand."""
        if self.config.stability_repeats == 0:
            return ()
        params = estimator.get_params(deep=False)
        minimum = max(
            int(params.get("n_clusters", 1)),
            int(params.get("n_components", 1)),
            3,
        )
        sample_size = min(
            len(training_features),
            max(minimum, int(math.ceil(0.8 * len(training_features)))),
        )
        rng = np.random.default_rng(self.config.random_seed)
        scores: list[float] = []
        for repeat in range(self.config.stability_repeats):
            indices = np.sort(
                rng.choice(len(training_features), size=sample_size, replace=False)
            )
            seed = (self.config.random_seed + repeat + 1) % (2**32)
            candidate = self._build_estimator(seed, sample_size)
            try:
                candidate.fit(training_features[indices])
                sampled_labels = self._cluster_labels(candidate, training_features[indices])
            except (ValueError, TypeError, FloatingPointError) as exc:
                raise MLTrainingError("cluster stability resampling failed") from exc
            score = float(adjusted_rand_score(training_labels[indices], sampled_labels))
            if not math.isfinite(score):
                raise MLTrainingError("cluster stability produced a non-finite score")
            scores.append(score)
        return tuple(scores)

    def fit(
        self,
        features: Sequence[Sequence[float]],
        feature_names: Sequence[str],
    ) -> UnsupervisedMLResult:
        """Fit on the initial prefix; only transform/score supported holdout rows."""
        matrix = _as_matrix(features, max_observations=self.config.max_observations)
        names = tuple(feature_names)
        if len(names) != matrix.shape[1]:
            raise MLInputError("feature_names must align with the number of feature columns")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise MLInputError("feature names must be non-empty strings")
        if len(set(names)) != len(names):
            raise MLInputError("feature names must be unique")

        train_end = int(math.floor(matrix.shape[0] * self.config.fit_fraction))
        if train_end < 8 or matrix.shape[0] - train_end < 2:
            raise MLInputError("split must leave at least eight training and two holdout rows")

        raw_train = matrix[:train_end]
        raw_holdout = matrix[train_end:]
        scaler = StandardScaler()
        try:
            train = scaler.fit_transform(raw_train)
            holdout = scaler.transform(raw_holdout)
        except (ValueError, TypeError, FloatingPointError) as exc:
            raise MLTrainingError("training-prefix scaling failed") from exc

        estimator = self._build_estimator(self.config.random_seed, len(train))
        operation = _OPERATION_BY_ALGORITHM[self.config.algorithm]
        try:
            estimator.fit(train)
            training_labels: tuple[int, ...] | None = None
            holdout_labels: tuple[int, ...] | None = None
            training_embedding: tuple[tuple[float, ...], ...] | None = None
            holdout_embedding: tuple[tuple[float, ...], ...] | None = None
            training_scores: tuple[float, ...] | None = None
            holdout_scores: tuple[float, ...] | None = None
            training_flags: tuple[int, ...] | None = None
            holdout_flags: tuple[int, ...] | None = None
            stability_scores: tuple[float, ...] = ()
            stability_mean: float | None = None

            if operation == "clustering":
                fitted_labels = self._cluster_labels(estimator, train)
                training_labels = tuple(int(value) for value in fitted_labels.tolist())
                if hasattr(estimator, "predict"):
                    predicted = np.asarray(estimator.predict(holdout), dtype=np.int64)
                    if predicted.ndim != 1 or len(predicted) != len(holdout):
                        raise MLTrainingError("clustering estimator returned misaligned holdout labels")
                    holdout_labels = tuple(int(value) for value in predicted.tolist())
                stability_scores = self._stability(estimator, train, fitted_labels)
                if stability_scores:
                    stability_mean = float(np.mean(stability_scores))
            elif operation == "dimensionality_reduction":
                train_embedding_array = np.asarray(estimator.transform(train), dtype=np.float64)
                holdout_embedding_array = np.asarray(estimator.transform(holdout), dtype=np.float64)
                if (
                    train_embedding_array.ndim != 2
                    or holdout_embedding_array.ndim != 2
                    or not np.all(np.isfinite(train_embedding_array))
                    or not np.all(np.isfinite(holdout_embedding_array))
                ):
                    raise MLTrainingError("dimensionality reduction returned invalid embeddings")
                training_embedding = tuple(
                    tuple(float(value) for value in row)
                    for row in train_embedding_array.tolist()
                )
                holdout_embedding = tuple(
                    tuple(float(value) for value in row)
                    for row in holdout_embedding_array.tolist()
                )
            else:
                if self.config.algorithm == "local_outlier_factor":
                    negative_factors = np.asarray(estimator.negative_outlier_factor_, dtype=np.float64)
                    training_scores_array = -negative_factors
                    offset = float(estimator.offset_)
                    training_flags_array = np.where(negative_factors < offset, -1, 1)
                else:
                    training_scores_array = -np.asarray(
                        estimator.score_samples(train), dtype=np.float64
                    )
                    training_flags_array = np.asarray(estimator.predict(train), dtype=np.int64)
                holdout_scores_array = -np.asarray(
                    estimator.score_samples(holdout), dtype=np.float64
                )
                holdout_flags_array = np.asarray(estimator.predict(holdout), dtype=np.int64)
                if (
                    not np.all(np.isfinite(training_scores_array))
                    or not np.all(np.isfinite(holdout_scores_array))
                ):
                    raise MLTrainingError("anomaly estimator returned non-finite scores")
                training_scores = tuple(float(value) for value in training_scores_array.tolist())
                holdout_scores = tuple(float(value) for value in holdout_scores_array.tolist())
                training_flags = tuple(int(value) for value in training_flags_array.tolist())
                holdout_flags = tuple(int(value) for value in holdout_flags_array.tolist())
        except MLInputError:
            raise
        except MLTrainingError:
            raise
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise MLTrainingError(
                f"unsupervised fitting failed for {self.config.algorithm}"
            ) from exc

        fingerprint = _fingerprint(raw_train, names)
        identity = json.dumps(
            {
                "algorithm": self.config.algorithm,
                "seed": self.config.random_seed,
                "parameters": self.config.parameters,
                "fingerprint": fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        parameters = _primitive_parameters(estimator)
        parameters["random_seed"] = self.config.random_seed
        parameters["fit_fraction"] = self.config.fit_fraction
        parameters["stability_repeats"] = self.config.stability_repeats
        parameters["scaler"] = "StandardScaler"
        parameters["scaler_fit_scope"] = "training_prefix_only"

        limitations = [
            "Clusters, components and anomaly flags are exploratory outputs, not causal findings or trading signals.",
            "Only the training prefix fits the scaler and estimator; row order must be supplied correctly by the caller.",
        ]
        if operation == "clustering" and holdout_labels is None:
            limitations.append(
                "This algorithm has no inductive prediction API; holdout cluster labels are intentionally omitted."
            )
        if operation == "clustering":
            limitations.append(
                "Adjusted Rand stability is a resampling diagnostic, not a guarantee of semantic cluster validity."
            )
        elif operation == "dimensionality_reduction":
            limitations.append(
                "Embedding axes are not automatically economically interpretable."
            )
        else:
            limitations.append(
                "Anomaly scores are relative to the fitted training distribution; larger scores indicate more unusual rows."
            )

        result = UnsupervisedMLResult(
            model_id=f"{self.config.algorithm}:{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
            algorithm=self.config.algorithm,
            operation=operation,
            feature_names=names,
            training_observations=len(raw_train),
            holdout_observations=len(raw_holdout),
            total_observations=len(matrix),
            train_end_index=train_end,
            holdout_start_index=train_end,
            holdout_used_for_fit=False,
            scaler_fit_scope="training_prefix_only",
            scaler_mean=tuple(float(value) for value in scaler.mean_.tolist()),
            scaler_scale=tuple(float(value) for value in scaler.scale_.tolist()),
            random_seed=self.config.random_seed,
            library_version=sklearn.__version__,
            data_fingerprint=fingerprint,
            parameters=parameters,
            training_labels=training_labels,
            holdout_labels=holdout_labels,
            training_embedding=training_embedding,
            holdout_embedding=holdout_embedding,
            training_anomaly_scores=training_scores,
            holdout_anomaly_scores=holdout_scores,
            training_anomaly_flags=training_flags,
            holdout_anomaly_flags=holdout_flags,
            stability_scores=stability_scores,
            stability_mean_ari=stability_mean,
            limitations=tuple(limitations),
        )
        self._estimator = estimator
        self._scaler = scaler
        self._feature_names = names
        self.result = result
        return result
