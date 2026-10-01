"""Contracts for deterministic return and performance calculations."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .market import Asset

ReturnType = Literal["simple", "log"]


class ReturnSeries(BaseModel):
    """Immutable time-indexed return series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    timestamps: tuple[datetime, ...]
    values: tuple[float, ...]
    frequency: str = Field(min_length=1)
    return_type: ReturnType

    @model_validator(mode="after")
    def validate_alignment(self) -> ReturnSeries:
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have equal length")
        if any(left >= right for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("return values must be finite")
        if self.return_type == "simple" and any(value <= -1.0 for value in self.values):
            raise ValueError("simple returns must be greater than -100%")
        return self

    @property
    def observations(self) -> int:
        """Number of return observations."""
        return len(self.values)


class PerformanceMetrics(BaseModel):
    """Deterministic performance metrics for a simple-return series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: int = Field(ge=1)
    periods_per_year: float = Field(gt=0)
    total_return: float
    annualized_return: float
    annualized_volatility: float = Field(ge=0)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float = Field(ge=0, le=1)
    positive_period_ratio: float = Field(ge=0, le=1)
