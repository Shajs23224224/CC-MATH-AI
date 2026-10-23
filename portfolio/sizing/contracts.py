"""Typed contracts for F37 auditable position sizing."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from portfolio.optimization.contracts import PortfolioOptimizationResult

_VERSION_PATTERN = re.compile(r"^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)(?:-[0-9A-Za-z.-]+)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

SizingMethod = Literal[
    "fixed_fractional",
    "volatility_based",
    "risk_based",
    "kelly",
    "fractional_kelly",
    "confidence_weighting",
]
SizingStatus = Literal["SIZED", "PARTIAL", "ZERO_SIZE"]
PositionStatus = Literal["SIZED", "ZERO_SIZE"]


class PositionSizingConfig(BaseModel):
    """Versioned policy; every parameter is expressed as a dimensionless fraction unless stated."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    config_id: str = Field(default="position-sizer", min_length=1, max_length=160)
    version: str = "1.0.0"
    method: SizingMethod = "risk_based"
    fixed_fractional_weight: float = Field(default=0.05, gt=0.0, le=1.0)
    risk_budget_fraction: float = Field(default=0.01, gt=0.0, le=0.25)
    volatility_budget_fraction: float = Field(default=0.02, gt=0.0, le=1.0)
    fractional_kelly_multiplier: float = Field(default=0.25, gt=0.0, le=1.0)
    risk_free_rate: float = Field(default=0.0, ge=-1.0, le=10.0)
    maximum_position_weight: float = Field(default=0.25, gt=0.0, le=1.0)
    max_participation_rate: float = Field(default=0.05, gt=0.0, le=0.5)
    minimum_trade_notional: float = Field(default=0.0, ge=0.0)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Require a semantic version so policies can be replayed."""
        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("version must use semantic versioning, e.g. 1.0.0")
        return value


class SizingAssetInput(BaseModel):
    """Point-in-time market inputs for one F36-approved asset."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    price: float = Field(gt=0.0)
    expected_return: float
    volatility: float = Field(ge=0.0, le=10.0, description="Return standard deviation over horizon")
    confidence: float = Field(ge=0.0, le=1.0)
    average_daily_volume: float = Field(gt=0.0, description="Average daily volume in shares/units")
    lot_size: float = Field(default=1.0, gt=0.0)
    stop_loss_price: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_stop_loss(self) -> SizingAssetInput:
        """Long-only stop losses must be below the current entry price."""
        if self.stop_loss_price is not None and self.stop_loss_price >= self.price:
            raise ValueError("stop_loss_price must be below price for long-only sizing")
        return self


class PositionSizingRequest(BaseModel):
    """A successful F36 portfolio output plus aligned market data and account equity."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    portfolio_optimization: PortfolioOptimizationResult
    assets: tuple[SizingAssetInput, ...] = Field(min_length=2, max_length=500)
    equity: float = Field(gt=0.0)
    base_currency: str = Field(min_length=3, max_length=3)
    as_of: datetime
    market_data_as_of: datetime
    horizon: str = Field(min_length=1, max_length=80)
    market_data_snapshot_id: str = Field(min_length=1, max_length=200)

    @field_validator("base_currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize currency labels for deterministic reporting."""
        normalized = value.strip().upper()
        if not normalized.isalpha() or len(normalized) != 3:
            raise ValueError("base_currency must be a three-letter currency code")
        return normalized

    @field_validator("as_of", "market_data_as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject ambiguous timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("sizing timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> PositionSizingRequest:
        """Reject invalid upstream allocations and temporal/universe mismatches."""
        optimization = self.portfolio_optimization
        if optimization.status not in {"OPTIMAL", "FEASIBLE"}:
            raise ValueError("F37 requires an OPTIMAL or FEASIBLE F36 result")
        if any(not item.satisfied for item in optimization.constraints):
            raise ValueError("F37 cannot consume an F36 result with failed constraints")
        if optimization.as_of > self.as_of:
            raise ValueError("portfolio optimization timestamp cannot be in the future")
        if self.market_data_as_of > self.as_of:
            raise ValueError("market_data_as_of cannot be later than sizing as_of")
        if optimization.horizon != self.horizon:
            raise ValueError("F36 optimization and F37 sizing horizons must match")
        symbols = tuple(asset.asset_symbol for asset in self.assets)
        if symbols != optimization.asset_symbols:
            raise ValueError("sizing assets must exactly match F36 symbol order")
        if len(set(symbols)) != len(symbols):
            raise ValueError("sizing asset symbols must be unique")
        if math.fsum(item.target_weight for item in optimization.allocations) > 1.0 + 1e-6:
            raise ValueError("F36 target weights must not exceed fully invested equity")
        return self


class SizedPosition(BaseModel):
    """Position quantity and audit diagnostics for one asset."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    source_target_weight: float = Field(ge=0.0, le=1.0)
    target_weight: float = Field(ge=0.0, le=1.0)
    target_quantity: float = Field(ge=0.0)
    target_notional: float = Field(ge=0.0)
    risk_budget_amount: float = Field(ge=0.0)
    estimated_loss_at_stop: float = Field(ge=0.0)
    adv_participation: float = Field(ge=0.0, le=1.0)
    liquidity_limited: bool
    position_weight_limited: bool
    status: PositionStatus
    rationale: str = Field(min_length=1, max_length=1500)

    @model_validator(mode="after")
    def validate_zero_consistency(self) -> SizedPosition:
        """A zero-sized line must not retain target exposure."""
        if self.target_quantity == 0.0:
            if self.target_weight != 0.0 or self.target_notional != 0.0:
                raise ValueError("zero quantity must have zero target weight and notional")
            if self.status != "ZERO_SIZE":
                raise ValueError("zero quantity must use ZERO_SIZE status")
        elif self.status != "SIZED":
            raise ValueError("positive quantity must use SIZED status")
        if self.target_weight > self.source_target_weight + 1e-8:
            raise ValueError("sizing cannot exceed the upstream F36 target weight")
        return self


class SizingResult(BaseModel):
    """Auditable batch output from F37; does not authorize orders."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    status: SizingStatus
    method: SizingMethod
    config_id: str = Field(min_length=1, max_length=160)
    config_version: str
    allocations: tuple[SizedPosition, ...] = Field(min_length=2, max_length=500)
    total_target_weight: float = Field(ge=0.0, le=1.0)
    total_target_notional: float = Field(ge=0.0)
    total_risk_budget_amount: float = Field(ge=0.0)
    equity: float = Field(gt=0.0)
    base_currency: str = Field(min_length=3, max_length=3)
    as_of: datetime
    horizon: str = Field(min_length=1, max_length=80)
    market_data_snapshot_id: str = Field(min_length=1, max_length=200)
    portfolio_optimization_fingerprint: str
    sizing_fingerprint: str

    @field_validator("as_of")
    @classmethod
    def require_result_timezone(cls, value: datetime) -> datetime:
        """Keep result time explicit."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @field_validator("portfolio_optimization_fingerprint", "sizing_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Require canonical SHA-256 identifiers."""
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("fingerprints must be lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def validate_batch_consistency(self) -> SizingResult:
        """Verify aggregate diagnostics and position statuses."""
        weights = math.fsum(item.target_weight for item in self.allocations)
        notionals = math.fsum(item.target_notional for item in self.allocations)
        budgets = math.fsum(item.risk_budget_amount for item in self.allocations)
        if not math.isclose(weights, self.total_target_weight, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("total_target_weight must equal the sum of position weights")
        if not math.isclose(notionals, self.total_target_notional, rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError("total_target_notional must equal the sum of position notionals")
        if not math.isclose(budgets, self.total_risk_budget_amount, rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError("total_risk_budget_amount must equal the sum of risk budgets")
        if weights > 1.0 + 1e-8:
            raise ValueError("F37 target weights must not exceed total equity")
        positive = sum(item.target_quantity > 0.0 for item in self.allocations)
        if self.status == "SIZED" and positive != len(self.allocations):
            raise ValueError("SIZED result requires every position to have positive quantity")
        if self.status == "PARTIAL" and not (0 < positive < len(self.allocations)):
            raise ValueError("PARTIAL result requires both sized and zero-sized positions")
        if self.status == "ZERO_SIZE" and positive != 0:
            raise ValueError("ZERO_SIZE result cannot contain positive quantities")
        return self


__all__ = [
    "PositionSizingConfig",
    "PositionSizingRequest",
    "SizingAssetInput",
    "SizingMethod",
    "SizingResult",
    "SizedPosition",
]
