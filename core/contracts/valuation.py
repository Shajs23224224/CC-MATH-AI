from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ValuationBasis = Literal["enterprise", "equity"]
ValuationUnit = Literal["currency", "currency_per_share", "multiple"]


class ValuationAssumptions(BaseModel):
    """Versioned assumptions shared by valuation models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assumptions_version: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    discount_rate: float | None = None
    terminal_growth_rate: float | None = None
    tax_rate: float | None = None

    @model_validator(mode="after")
    def validate_rates(self) -> ValuationAssumptions:
        for name in ("discount_rate", "terminal_growth_rate", "tax_rate"):
            value = getattr(self, name)
            if value is not None and value != value:
                raise ValueError(f"{name} must be finite")
        if self.discount_rate is not None and not 0.0 < self.discount_rate < 1.0:
            raise ValueError("discount_rate must be between 0 and 1")
        if self.terminal_growth_rate is not None and not -1.0 < self.terminal_growth_rate < 1.0:
            raise ValueError("terminal_growth_rate must be between -1 and 1")
        if self.tax_rate is not None and not -1.0 < self.tax_rate < 1.0:
            raise ValueError("tax_rate must be between -1 and 1")
        return self


class ValuationResult(BaseModel):
    """Immutable valuation output with explicit units and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(min_length=1)
    assumptions_version: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    basis: ValuationBasis
    unit: ValuationUnit
    value: float
    enterprise_value: float | None = None
    equity_value: float | None = None
    per_share_value: float | None = None

    @model_validator(mode="after")
    def validate_value(self) -> ValuationResult:
        import math

        if not math.isfinite(self.value):
            raise ValueError("valuation value must be finite")
        for name in ("enterprise_value", "equity_value", "per_share_value"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.unit == "currency_per_share" and self.per_share_value is None:
            raise ValueError("per_share_value is required for currency_per_share")
        return self


class SensitivityMatrix(BaseModel):
    """Two-dimensional valuation sensitivity table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(min_length=1)
    assumptions_version: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    row_parameter: str = Field(min_length=1)
    column_parameter: str = Field(min_length=1)
    row_values: tuple[float, ...]
    column_values: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def validate_shape(self) -> SensitivityMatrix:
        if not self.row_values or not self.column_values:
            raise ValueError("sensitivity axes cannot be empty")
        if len(self.values) != len(self.row_values):
            raise ValueError("sensitivity row count does not match row_values")
        if any(len(row) != len(self.column_values) for row in self.values):
            raise ValueError("sensitivity column count does not match column_values")
        return self
