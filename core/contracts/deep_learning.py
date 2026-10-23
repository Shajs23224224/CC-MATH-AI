"""Typed, bounded contracts for reproducible temporal deep learning."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DeepLearningAlgorithm = Literal[
    "mlp",
    "lstm",
    "gru",
    "tcn",
    "autoencoder",
    "vae",
    "attention",
    "transformer",
    "temporal_fusion_transformer",
]
DeepLearningTask = Literal["regression", "classification", "forecasting", "reconstruction"]


class DeepLearningConfig(BaseModel):
    """Allow-listed PyTorch architecture and resource limits for F31."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    algorithm: DeepLearningAlgorithm
    task: DeepLearningTask
    random_seed: int = Field(default=31, ge=0, le=2**32 - 1)
    test_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    validation_fraction: float = Field(default=0.15, gt=0.05, lt=0.3)
    sequence_length: int = Field(default=8, ge=1, le=128)
    hidden_dim: int = Field(default=32, ge=4, le=256)
    num_layers: int = Field(default=2, ge=1, le=4)
    num_heads: int = Field(default=4, ge=1, le=16)
    dropout: float = Field(default=0.1, ge=0.0, le=0.8)
    learning_rate: float = Field(default=0.001, gt=0.00001, lt=0.1)
    weight_decay: float = Field(default=0.0001, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1, le=512)
    epochs: int = Field(default=100, ge=1, le=200)
    patience: int = Field(default=12, ge=1, le=40)
    max_observations: int = Field(default=10_000, ge=60, le=50_000)
    max_parameters: int = Field(default=500_000, ge=1_000, le=1_000_000)
    device: Literal["cpu", "cuda"] = "cpu"

    @model_validator(mode="after")
    def validate_architecture_and_budget(self) -> DeepLearningConfig:
        """Reject incompatible task/architecture combinations and invalid time splits."""
        if self.test_fraction + self.validation_fraction >= 0.6:
            raise ValueError("test_fraction + validation_fraction must be below 0.6")
        reconstruction_model = self.algorithm in {"autoencoder", "vae"}
        if reconstruction_model != (self.task == "reconstruction"):
            raise ValueError(
                "autoencoder and vae require reconstruction; other architectures require "
                "regression, classification or forecasting"
            )
        if self.algorithm in {"attention", "transformer", "temporal_fusion_transformer"}:
            if self.hidden_dim % self.num_heads:
                raise ValueError("hidden_dim must be divisible by num_heads")
        return self


class DeepLearningResult(BaseModel):
    """Portable fit report; learned weights are not embedded in this metadata contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    algorithm: DeepLearningAlgorithm
    task: DeepLearningTask
    feature_names: tuple[str, ...] = Field(min_length=1, max_length=64)
    target_name: str = Field(min_length=1)
    total_observations: int = Field(ge=12)
    training_samples: int = Field(ge=8)
    validation_samples: int = Field(ge=2)
    test_samples: int = Field(ge=2)
    train_end_index: int = Field(ge=8)
    validation_start_index: int = Field(ge=8)
    validation_end_index: int = Field(ge=10)
    test_start_index: int = Field(ge=10)
    test_end_index: int = Field(ge=12)
    sequence_length: int = Field(ge=1, le=128)
    split_method: Literal["chronological_train_validation_test"] = (
        "chronological_train_validation_test"
    )
    validation_used_for_early_stopping: Literal[True] = True
    test_used_for_selection: Literal[False] = False
    scaler_fit_scope: Literal["training_features_only"] = "training_features_only"
    scaler_fit_end_index: int = Field(ge=1)
    scaler_mean: tuple[float, ...] = Field(min_length=1)
    scaler_scale: tuple[float, ...] = Field(min_length=1)
    random_seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    library_version: str = Field(min_length=1)
    data_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_count: int = Field(ge=1, le=1_000_000)
    epochs_requested: int = Field(ge=1, le=200)
    epochs_completed: int = Field(ge=1, le=200)
    best_epoch: int = Field(ge=1, le=200)
    early_stopped: bool
    best_validation_loss: float = Field(ge=0.0)
    metrics: dict[str, float]
    test_predictions: tuple[float | str, ...] = ()
    test_targets: tuple[float | str, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_temporal_boundaries_and_outputs(self) -> DeepLearningResult:
        """Guarantee contiguous splits, bounded finite metrics and aligned test outputs."""
        if self.validation_start_index != self.train_end_index:
            raise ValueError("validation must start where training ends")
        if self.test_start_index != self.validation_end_index:
            raise ValueError("test must start where validation ends")
        if not (
            self.train_end_index < self.validation_end_index < self.test_end_index
            and self.test_end_index == self.total_observations
        ):
            raise ValueError("chronological split indices must be strictly increasing")
        if self.training_samples != self.train_end_index - self.sequence_length:
            raise ValueError("training sample count must match sequence-window boundaries")
        if self.validation_samples != self.validation_end_index - self.validation_start_index:
            raise ValueError("validation sample count must match validation boundaries")
        if self.test_samples != self.test_end_index - self.test_start_index:
            raise ValueError("test sample count must match test boundaries")
        if len(self.feature_names) != len(self.scaler_mean) or len(self.feature_names) != len(
            self.scaler_scale
        ):
            raise ValueError("scaler statistics must align with feature_names")
        if any(value <= 0.0 for value in self.scaler_scale):
            raise ValueError("scaler scales must be positive")
        if any(not math.isfinite(value) for value in (*self.scaler_mean, *self.scaler_scale)):
            raise ValueError("scaler statistics must be finite")
        if not 1 <= self.best_epoch <= self.epochs_completed <= self.epochs_requested:
            raise ValueError("best_epoch and epochs_completed must fit within the requested budget")
        if not math.isfinite(self.best_validation_loss):
            raise ValueError("best validation loss must be finite")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("metrics must be finite")
        if self.task == "reconstruction":
            if self.test_predictions or self.test_targets:
                raise ValueError(
                    "reconstruction uses reconstruction metrics, not scalar target rows"
                )
            if "reconstruction_mse" not in self.metrics:
                raise ValueError("reconstruction requires reconstruction_mse")
        else:
            if len(self.test_predictions) != self.test_samples:
                raise ValueError("test_predictions must align with the test segment")
            if len(self.test_targets) != self.test_samples:
                raise ValueError("test_targets must align with the test segment")
        return self
