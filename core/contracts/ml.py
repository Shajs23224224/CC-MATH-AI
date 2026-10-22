"""Typed contracts for reproducible classical ML and bounded discovery."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

MLTask = Literal["regression", "classification"]
CandidateStatus = Literal[
    "CANDIDATE", "EXPERIMENTAL", "VALIDATED", "REJECTED", "DEPRECATED"
]


class MLDataset(BaseModel):
    """Immutable, ordered supervised dataset; row order is meaningful for temporal evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    dataset_id: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=500)
    feature_names: tuple[str, ...] = Field(min_length=1, max_length=64)
    target_name: str = Field(min_length=1, max_length=200)
    features: tuple[tuple[float, ...], ...] = Field(min_length=12)
    target: tuple[float | str, ...] = Field(min_length=12)
    task: MLTask

    @model_validator(mode="after")
    def validate_dataset(self) -> MLDataset:
        """Reject misaligned, ragged, non-finite or semantically inconsistent data."""
        if len(self.features) != len(self.target):
            raise ValueError("features and target must have the same number of rows")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if any(not name.strip() for name in self.feature_names):
            raise ValueError("feature_names must not be blank")
        width = len(self.feature_names)
        if any(len(row) != width for row in self.features):
            raise ValueError("feature rows must match feature_names")
        if any(not math.isfinite(value) for row in self.features for value in row):
            raise ValueError("features must contain only finite values")
        if self.task == "regression":
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in self.target
            ):
                raise ValueError("regression targets must be finite numeric values")
        else:
            if any(not isinstance(value, (int, float, str)) for value in self.target):
                raise ValueError("classification targets must be numeric or strings")
            if all(isinstance(value, str) for value in self.target):
                if any(not value for value in self.target):
                    raise ValueError("classification labels must not be empty")
            elif not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in self.target
            ):
                raise ValueError("classification targets must use one consistent label type")
            if len(set(self.target)) < 2:
                raise ValueError("classification requires at least two target classes")
        return self


class MLTrainingConfig(BaseModel):
    """Allow-listed estimator settings and chronological holdout configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    algorithm: str = Field(min_length=1, max_length=80)
    task: MLTask
    random_seed: int = Field(default=28, ge=0, le=2**32 - 1)
    test_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    parameters: dict[str, int | float | str | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> MLTrainingConfig:
        if "random_state" in self.parameters:
            raise ValueError("random_state is controlled by random_seed")
        for key, value in self.parameters.items():
            if not key or not key.isidentifier():
                raise ValueError("parameter names must be Python identifiers")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("numeric model parameters must be finite")
        return self


class MLModelMetadata(BaseModel):
    """Portable lineage metadata; does not serialize executable model objects."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    task: MLTask
    feature_names: tuple[str, ...] = Field(min_length=1)
    training_observations: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    library_version: str = Field(min_length=1)
    data_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: dict[str, int | float | str | bool | None]


class MLPrediction(BaseModel):
    """Prediction output with optional class-probability rows."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    predictions: tuple[float | str, ...] = Field(min_length=1)
    classes: tuple[float | str, ...] | None = None
    probabilities: tuple[tuple[float, ...], ...] | None = None

    @model_validator(mode="after")
    def validate_probabilities(self) -> MLPrediction:
        if self.probabilities is None:
            if self.classes is not None:
                raise ValueError("classes require probabilities")
            return self
        if self.classes is None or not self.classes:
            raise ValueError("probabilities require class labels")
        if len(self.probabilities) != len(self.predictions):
            raise ValueError("probability rows must align with predictions")
        for row in self.probabilities:
            if len(row) != len(self.classes):
                raise ValueError("each probability row must align with classes")
            if any(not 0.0 <= value <= 1.0 for value in row):
                raise ValueError("class probabilities must lie in [0, 1]")
            if not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("class probabilities must sum to one")
        return self


class MLEvaluation(BaseModel):
    """Chronological out-of-sample metrics; test observations are never used to select models."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    algorithm: str = Field(min_length=1)
    task: MLTask
    training_samples: int = Field(ge=1)
    test_samples: int = Field(ge=1)
    train_end_index: int = Field(ge=1)
    test_start_index: int = Field(ge=1)
    test_end_index: int = Field(ge=1)
    split_method: Literal["chronological"] = "chronological"
    test_used_for_selection: Literal[False] = False
    metrics: dict[str, float]

    @model_validator(mode="after")
    def validate_split_and_metrics(self) -> MLEvaluation:
        if self.train_end_index != self.test_start_index:
            raise ValueError("chronological train/test boundaries must be contiguous")
        if self.test_end_index - self.test_start_index != self.test_samples:
            raise ValueError("test_end_index must match test_samples")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("evaluation metrics must be finite")
        return self


class PatternRecord(BaseModel):
    """Exploratory association record that explicitly disclaims causal interpretation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    pattern_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    pattern_type: Literal["linear_association"] = "linear_association"
    feature_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    observations: int = Field(ge=4)
    pearson_correlation: float = Field(ge=-1.0, le=1.0)
    spearman_correlation: float = Field(ge=-1.0, le=1.0)
    stability_delta: float = Field(ge=0.0, le=2.0)
    hypothesis: str = Field(min_length=1)
    interpretation: Literal["association_only"] = "association_only"
    status: CandidateStatus = "EXPERIMENTAL"
    limitations: tuple[str, ...] = Field(min_length=1)


class SymbolicRegressionResult(BaseModel):
    """Polynomial candidate selected on validation and reported on an untouched test tail."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    candidate_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    algorithm: Literal["bounded_polynomial_basis_search"] = "bounded_polynomial_basis_search"
    status: CandidateStatus = "EXPERIMENTAL"
    feature_names: tuple[str, ...] = Field(min_length=1)
    degree: int = Field(ge=1, le=4)
    basis_terms: tuple[str, ...] = Field(min_length=1)
    coefficients: tuple[float, ...] = Field(min_length=1)
    intercept: float
    expression: str = Field(min_length=1)
    complexity: int = Field(ge=1)
    training_samples: int = Field(ge=1)
    validation_samples: int = Field(ge=1)
    test_samples: int = Field(ge=1)
    train_end_index: int = Field(ge=1)
    validation_end_index: int = Field(ge=1)
    test_start_index: int = Field(ge=1)
    validation_rmse: float = Field(ge=0.0)
    test_rmse: float = Field(ge=0.0)
    test_mae: float = Field(ge=0.0)
    test_r2: float
    test_used_for_selection: Literal[False] = False
    max_degree: int = Field(ge=1, le=4)
    max_terms: int = Field(ge=1, le=512)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> SymbolicRegressionResult:
        if len(self.basis_terms) != len(self.coefficients):
            raise ValueError("basis_terms and coefficients must align")
        if self.degree > self.max_degree:
            raise ValueError("selected degree cannot exceed max_degree")
        if self.train_end_index > self.validation_end_index:
            raise ValueError("validation boundary cannot precede training boundary")
        if self.test_start_index != self.validation_end_index:
            raise ValueError("test must begin directly after validation")
        if any(not math.isfinite(value) for value in self.coefficients):
            raise ValueError("coefficients must be finite")
        return self


class ExperimentRecord(BaseModel):
    """Versioned, non-executable experiment ledger record."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    experiment_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: str = Field(min_length=1, max_length=100)
    subject_id: str = Field(min_length=1, max_length=200)
    status: CandidateStatus
    configuration: dict[str, str | int | float | bool]
    results: dict[str, str | int | float | bool]
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_timestamp_and_values(self) -> ExperimentRecord:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        for values in (self.configuration, self.results):
            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in values.values()
            ):
                raise ValueError("experiment values must be finite")
        return self
