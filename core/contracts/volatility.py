"""Contracts for volatility and stochastic-process modelling."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VolatilityFamily = Literal[
    "arch",
    "garch",
    "egarch",
    "gjr_garch",
    "stochastic_volatility",
    "dcc_garch",
]

StochasticProcessFamily = Literal["ornstein_uhlenbeck", "mean_reversion"]


class VolatilityFitSummary(BaseModel):
    """Portable fit metadata for conditional-volatility models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: VolatilityFamily
    observations: int = Field(ge=2)
    parameter_count: int = Field(ge=0)
    log_likelihood: float
    aic: float
    bic: float
    converged: bool

    @model_validator(mode="after")
    def validate_finite(self) -> VolatilityFitSummary:
        for name in ("log_likelihood", "aic", "bic"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        return self


class VolatilityForecast(BaseModel):
    """Out-of-sample conditional volatility forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: VolatilityFamily
    horizon: int = Field(gt=0)
    annualization_factor: float = Field(gt=0)
    values: tuple[float, ...]

    @model_validator(mode="after")
    def validate_values(self) -> VolatilityForecast:
        if len(self.values) != self.horizon:
            raise ValueError("volatility forecast values must match horizon")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.values):
            raise ValueError("volatility forecast values must be finite and non-negative")
        return self


class DynamicCorrelationForecast(BaseModel):
    """Forecasted dynamic correlation matrix from DCC-GARCH."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Literal["dcc_garch"]
    horizon: int = Field(gt=0)
    matrix: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> DynamicCorrelationForecast:
        size = len(self.matrix)
        if size < 2 or any(len(row) != size for row in self.matrix):
            raise ValueError("correlation matrix must be square with at least two series")
        for row in self.matrix:
            for value in row:
                if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                    raise ValueError("correlation matrix entries must be finite and bounded")
        for i in range(size):
            if abs(self.matrix[i][i] - 1.0) > 1e-8:
                raise ValueError("correlation matrix diagonal must equal one")
            for j in range(size):
                if abs(self.matrix[i][j] - self.matrix[j][i]) > 1e-8:
                    raise ValueError("correlation matrix must be symmetric")
        return self


class MeanReversionFit(BaseModel):
    """Estimated continuous-time mean-reversion parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: StochasticProcessFamily
    observations: int = Field(ge=3)
    long_run_mean: float
    speed: float = Field(gt=0)
    diffusion: float = Field(gt=0)
    half_life: float = Field(gt=0)
    residual_std: float = Field(ge=0)
    converged: bool

    @model_validator(mode="after")
    def validate_finite(self) -> MeanReversionFit:
        for name in (
            "long_run_mean",
            "speed",
            "diffusion",
            "half_life",
            "residual_std",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        return self


class MeanReversionForecast(BaseModel):
    """Deterministic conditional mean path for a mean-reverting process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: StochasticProcessFamily
    horizon: int = Field(gt=0)
    values: tuple[float, ...]

    @model_validator(mode="after")
    def validate_values(self) -> MeanReversionForecast:
        if len(self.values) != self.horizon:
            raise ValueError("mean-reversion forecast values must match horizon")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("mean-reversion forecast values must be finite")
        return self
