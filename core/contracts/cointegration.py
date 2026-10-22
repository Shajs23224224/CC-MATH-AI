"""Typed contracts for F24 unit-root, cointegration and pairs-statistics results."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

UnitRootTestName = Literal["adf", "kpss", "phillips_perron"]


class UnitRootTestResult(BaseModel):
    """Auditable unit-root or stationarity test output."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    test: UnitRootTestName
    statistic: float
    pvalue: float = Field(ge=0.0, le=1.0)
    lags: int = Field(ge=0)
    observations: int = Field(ge=20)
    trend: Literal["c", "ct"]
    critical_values: dict[str, float] = Field(min_length=1)
    null_hypothesis: str = Field(min_length=1)
    alternative_hypothesis: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_critical_values(self) -> UnitRootTestResult:
        """Require finite critical values for documented significance levels."""
        if not all(math.isfinite(value) for value in self.critical_values.values()):
            raise ValueError("critical values must be finite")
        if not {"1%", "5%", "10%"}.issubset(self.critical_values):
            raise ValueError("critical values must include 1%, 5% and 10% levels")
        return self


class CointegrationTestResult(BaseModel):
    """Engle-Granger test output for the null of no cointegration."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    test: Literal["engle_granger"] = "engle_granger"
    statistic: float
    pvalue: float = Field(ge=0.0, le=1.0)
    observations: int = Field(ge=20)
    trend: Literal["c", "ct", "ctt", "n"]
    critical_values: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_results(self) -> CointegrationTestResult:
        """Reject non-finite test statistics or critical values."""
        if not math.isfinite(self.statistic) or not all(
            math.isfinite(value) for value in self.critical_values
        ):
            raise ValueError("cointegration statistics and critical values must be finite")
        return self


class JohansenTestResult(BaseModel):
    """Johansen trace and maximum-eigenvalue test results."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    assets: tuple[str, ...] = Field(min_length=2)
    observations: int = Field(ge=20)
    det_order: int = Field(ge=-1, le=1)
    k_ar_diff: int = Field(ge=0)
    eigenvalues: tuple[float, ...] = Field(min_length=2)
    trace_statistics: tuple[float, ...] = Field(min_length=2)
    max_eigen_statistics: tuple[float, ...] = Field(min_length=2)
    trace_critical_values: tuple[tuple[float, float, float], ...]
    max_eigen_critical_values: tuple[tuple[float, float, float], ...]
    trace_rank_95: int = Field(ge=0)
    max_eigen_rank_95: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_johansen(self) -> JohansenTestResult:
        """Validate matrix/vector dimensions and all finite numerical outputs."""
        size = len(self.assets)
        if len(set(self.assets)) != size or tuple(sorted(self.assets)) != self.assets:
            raise ValueError("Johansen asset symbols must be unique and sorted")
        if any(not symbol or symbol != symbol.strip().upper() for symbol in self.assets):
            raise ValueError("Johansen asset symbols must be non-empty uppercase labels")
        for values in (
            self.eigenvalues,
            self.trace_statistics,
            self.max_eigen_statistics,
        ):
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError("Johansen statistic vectors must be finite and match asset count")
        for matrix in (self.trace_critical_values, self.max_eigen_critical_values):
            if len(matrix) != size or any(
                len(row) != 3 or not all(math.isfinite(value) for value in row) for row in matrix
            ):
                raise ValueError("Johansen critical values must have shape (assets, 3)")
        if not all(0.0 <= value <= 1.0 + 1e-8 for value in self.eigenvalues):
            raise ValueError("Johansen eigenvalues must be between zero and one")
        if self.trace_rank_95 > size or self.max_eigen_rank_95 > size:
            raise ValueError("estimated cointegration rank cannot exceed asset count")
        return self


class CointegratingPairResult(BaseModel):
    """Pair regression, spread and Engle-Granger diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    dependent_symbol: str = Field(min_length=1)
    independent_symbol: str = Field(min_length=1)
    observations: int = Field(ge=20)
    intercept: float
    hedge_ratio: float
    spread: tuple[float, ...] = Field(min_length=20)
    cointegration_statistic: float
    cointegration_pvalue: float = Field(ge=0.0, le=1.0)
    critical_values: tuple[float, float, float]
    half_life: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_pair_result(self) -> CointegratingPairResult:
        """Keep the spread length and pair identity consistent with metadata."""
        if self.dependent_symbol.strip().upper() == self.independent_symbol.strip().upper():
            raise ValueError("pair symbols must be distinct")
        if len(self.spread) != self.observations:
            raise ValueError("spread length must equal observations")
        values = (
            self.intercept,
            self.hedge_ratio,
            self.cointegration_statistic,
            *self.critical_values,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pair parameters and critical values must be finite")
        if not all(math.isfinite(value) for value in self.spread):
            raise ValueError("spread values must be finite")
        return self


class MultipleTestingResult(BaseModel):
    """One Benjamini-Hochberg adjusted p-value, preserving input order."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    original_index: int = Field(ge=0)
    rank: int = Field(ge=1)
    pvalue: float = Field(ge=0.0, le=1.0)
    adjusted_pvalue: float = Field(ge=0.0, le=1.0)
    reject_null: bool
    alpha: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_decision(self) -> MultipleTestingResult:
        """Require the decision to match the adjusted p-value threshold."""
        if self.reject_null != (self.adjusted_pvalue <= self.alpha):
            raise ValueError("reject_null must match adjusted_pvalue <= alpha")
        return self
