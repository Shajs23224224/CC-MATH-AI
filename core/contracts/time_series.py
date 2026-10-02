"""Contracts for deterministic time-series modelling and diagnostics."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TimeSeriesFamily = Literal[
    "ar",
    "ma",
    "arma",
    "arima",
    "sarima",
    "arimax",
    "exponential_smoothing",
    "holt",
    "holt_winters",
    "var",
    "vecm",
]


class TimeSeriesForecast(BaseModel):
    """Validated out-of-sample forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: TimeSeriesFamily
    horizon: int = Field(gt=0)
    values: tuple[float, ...]

    @model_validator(mode="after")
    def validate_values(self) -> TimeSeriesForecast:
        if len(self.values) != self.horizon:
            raise ValueError("forecast values must match horizon")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("forecast values must be finite")
        return self


class TimeSeriesFitSummary(BaseModel):
    """Portable fit metadata independent of the underlying estimator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: TimeSeriesFamily
    observations: int = Field(ge=1)
    parameter_count: int = Field(ge=0)
    aic: float | None = None
    bic: float | None = None
    converged: bool


class ResidualDiagnostics(BaseModel):
    """Residual diagnostics for a fitted time-series model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: int = Field(ge=2)
    mean: float
    variance: float = Field(ge=0)
    lag: int = Field(ge=1)
    ljung_box_statistic: float = Field(ge=0)
    ljung_box_pvalue: float = Field(ge=0, le=1)
    residual_autocorrelation: float


class TemporalSplit(BaseModel):
    """A strict chronological train/test partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_size: int = Field(ge=1)
    test_size: int = Field(ge=1)
    total_size: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_sizes(self) -> TemporalSplit:
        if self.train_size + self.test_size != self.total_size:
            raise ValueError("train_size + test_size must equal total_size")
        return self
