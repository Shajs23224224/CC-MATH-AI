"""Typed contracts for leakage-aware unsupervised learning and diagnostics."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

UnsupervisedAlgorithm = Literal[
    "kmeans",
    "agglomerative_clustering",
    "dbscan",
    "gaussian_mixture",
    "pca",
    "kernel_pca",
    "fast_ica",
    "isolation_forest",
    "local_outlier_factor",
    "elliptic_envelope",
]
UnsupervisedOperation = Literal["clustering", "dimensionality_reduction", "anomaly_detection"]


class UnsupervisedMLConfig(BaseModel):
    """Bounded configuration for F30 unsupervised algorithms."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    algorithm: UnsupervisedAlgorithm
    random_seed: int = Field(default=30, ge=0, le=2**32 - 1)
    fit_fraction: float = Field(default=0.8, gt=0.5, lt=0.95)
    stability_repeats: int = Field(default=3, ge=0, le=5)
    max_observations: int = Field(default=5_000, ge=12, le=5_000)
    parameters: dict[str, int | float | str | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> UnsupervisedMLConfig:
        """Keep seeds and resource-heavy controls under the service's control."""
        if "random_state" in self.parameters:
            raise ValueError("random_state is controlled by random_seed")
        if "n_jobs" in self.parameters:
            raise ValueError("n_jobs is fixed to one for reproducible CPU-bounded execution")
        for key, value in self.parameters.items():
            if not key or not key.isidentifier():
                raise ValueError("parameter names must be Python identifiers")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("numeric model parameters must be finite")
            limits = {"n_estimators": 500, "n_init": 20, "max_iter": 1_000, "n_neighbors": 1_000}
            if key in limits and isinstance(value, int) and not isinstance(value, bool):
                if value > limits[key]:
                    raise ValueError(f"{key} cannot exceed {limits[key]}")
        return self


class UnsupervisedMLResult(BaseModel):
    """Auditable outputs; holdout rows never contribute to fitting or scaling."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model_id: str = Field(min_length=1)
    algorithm: UnsupervisedAlgorithm
    operation: UnsupervisedOperation
    feature_names: tuple[str, ...] = Field(min_length=1, max_length=64)
    training_observations: int = Field(ge=8)
    holdout_observations: int = Field(ge=2)
    total_observations: int = Field(ge=12)
    train_end_index: int = Field(ge=8)
    holdout_start_index: int = Field(ge=8)
    split_method: Literal["chronological_prefix_holdout"] = "chronological_prefix_holdout"
    holdout_used_for_fit: Literal[False] = False
    scaler_fit_scope: Literal["training_prefix_only"] = "training_prefix_only"
    scaler_mean: tuple[float, ...] = Field(min_length=1)
    scaler_scale: tuple[float, ...] = Field(min_length=1)
    random_seed: int = Field(ge=0)
    library_version: str = Field(min_length=1)
    data_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: dict[str, int | float | str | bool | None]
    training_labels: tuple[int, ...] | None = None
    holdout_labels: tuple[int, ...] | None = None
    training_embedding: tuple[tuple[float, ...], ...] | None = None
    holdout_embedding: tuple[tuple[float, ...], ...] | None = None
    training_anomaly_scores: tuple[float, ...] | None = None
    holdout_anomaly_scores: tuple[float, ...] | None = None
    training_anomaly_flags: tuple[int, ...] | None = None
    holdout_anomaly_flags: tuple[int, ...] | None = None
    stability_scores: tuple[float, ...] = ()
    stability_mean_ari: float | None = Field(default=None, ge=-1.0, le=1.0)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result_shapes(self) -> UnsupervisedMLResult:
        """Reject partial or misaligned output payloads."""
        if len(self.feature_names) != len(self.scaler_mean) or len(self.feature_names) != len(
            self.scaler_scale
        ):
            raise ValueError("scaler statistics must align with feature_names")
        if self.train_end_index != self.holdout_start_index:
            raise ValueError("training and holdout boundaries must be contiguous")
        if self.total_observations != self.training_observations + self.holdout_observations:
            raise ValueError("observation counts must sum to total_observations")
        if any(value <= 0.0 for value in self.scaler_scale):
            raise ValueError("scaler scales must be positive")
        if any(not math.isfinite(value) for value in (*self.scaler_mean, *self.scaler_scale)):
            raise ValueError("scaler statistics must be finite")
        if self.operation == "clustering":
            if (
                self.training_labels is None
                or len(self.training_labels) != self.training_observations
            ):
                raise ValueError("clustering results require one training label per training row")
            if (
                self.holdout_labels is not None
                and len(self.holdout_labels) != self.holdout_observations
            ):
                raise ValueError("holdout labels must align with holdout rows")
        elif self.operation == "dimensionality_reduction":
            if (
                self.training_embedding is None
                or len(self.training_embedding) != self.training_observations
            ):
                raise ValueError(
                    "reduction results require one training embedding per training row"
                )
            if (
                self.holdout_embedding is None
                or len(self.holdout_embedding) != self.holdout_observations
            ):
                raise ValueError("reduction results require one holdout embedding per holdout row")
            dimensions = len(self.training_embedding[0])
            if dimensions < 1 or any(len(row) != dimensions for row in self.training_embedding):
                raise ValueError("training embedding dimensions must be consistent")
            if any(len(row) != dimensions for row in self.holdout_embedding):
                raise ValueError("holdout embedding dimensions must match training embeddings")
        else:
            expected = self.training_observations
            if (
                self.training_anomaly_scores is None
                or len(self.training_anomaly_scores) != expected
            ):
                raise ValueError("anomaly results require one score per training row")
            if self.training_anomaly_flags is None or len(self.training_anomaly_flags) != expected:
                raise ValueError("anomaly results require one flag per training row")
            if (
                self.holdout_anomaly_scores is None
                or len(self.holdout_anomaly_scores) != self.holdout_observations
            ):
                raise ValueError("anomaly results require one score per holdout row")
            if (
                self.holdout_anomaly_flags is None
                or len(self.holdout_anomaly_flags) != self.holdout_observations
            ):
                raise ValueError("anomaly results require one flag per holdout row")
            if any(
                flag not in {-1, 1}
                for flag in (*self.training_anomaly_flags, *self.holdout_anomaly_flags)
            ):
                raise ValueError("anomaly flags must be 1 for inlier or -1 for outlier")
        if any(not -1.0 <= value <= 1.0 for value in self.stability_scores):
            raise ValueError("adjusted Rand stability scores must lie in [-1, 1]")
        if any(not math.isfinite(value) for value in self.stability_scores):
            raise ValueError("stability scores must be finite")
        return self
