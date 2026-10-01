"""Contracts for deterministic statistical analysis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StatisticalSummary(BaseModel):
    """Immutable descriptive-statistics summary for a numeric sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: int = Field(ge=1)
    mean: float
    median: float
    variance: float = Field(ge=0)
    standard_deviation: float = Field(ge=0)
    skewness: float
    excess_kurtosis: float
    minimum: float
    maximum: float
    first_quartile: float
    third_quartile: float
    interquartile_range: float = Field(ge=0)
    median_absolute_deviation: float = Field(ge=0)
