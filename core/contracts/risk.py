"""Typed contracts for deterministic portfolio-risk evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import RiskAction


class RiskMetric(BaseModel):
    """A named, auditable risk measure."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    methodology: str = Field(min_length=1)


class RiskConstraint(BaseModel):
    """A threshold applied to a named metric.

    The max operator breaches when the metric is above its limit; min breaches
    when the metric is below its limit. A hard breach blocks, a soft breach
    reduces the eligible allocation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    constraint_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    limit: float
    metric_name: str = Field(min_length=1)
    operator: Literal["max", "min"] = "max"
    hard: bool = True
    expected_unit: str | None = Field(default=None, min_length=1)


class RiskEvaluation(BaseModel):
    """Decision record emitted by the risk constraint evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    action: RiskAction
    reasons: tuple[str, ...] = ()
    metrics: tuple[RiskMetric, ...] = ()
    triggered_constraints: tuple[str, ...] = ()
    reduction_factor: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_action_factor(self) -> RiskEvaluation:
        """Keep the decision and its allocation multiplier internally consistent."""
        if self.action == RiskAction.ALLOW and self.reduction_factor != 1.0:
            raise ValueError("ALLOW requires reduction_factor=1.0")
        if self.action == RiskAction.BLOCK and self.reduction_factor != 0.0:
            raise ValueError("BLOCK requires reduction_factor=0.0")
        if self.action == RiskAction.REDUCE and not 0.0 < self.reduction_factor < 1.0:
            raise ValueError("REDUCE requires reduction_factor strictly between 0 and 1")
        return self


class PortfolioExposureMetrics(BaseModel):
    """Portfolio exposure measures calculated from signed market values."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    portfolio_id: str = Field(min_length=1)
    timestamp: datetime
    base_currency: str = Field(min_length=3, max_length=3)
    equity: float = Field(gt=0.0)
    gross_exposure: float = Field(ge=0.0)
    net_exposure: float
    gross_leverage: float = Field(ge=0.0)
    net_exposure_ratio: float
    cash_weight: float
    concentration_hhi: float = Field(ge=0.0, le=1.0)
    largest_position_weight: float = Field(ge=0.0, le=1.0)
    position_count: int = Field(ge=0)


class LiquidityRiskMetrics(BaseModel):
    """Estimated liquidation burden based on average daily dollar volume."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1)
    position_notional: float = Field(ge=0.0)
    average_daily_dollar_volume: float = Field(gt=0.0)
    max_participation_rate: float = Field(gt=0.0, le=1.0)
    participation_rate: float = Field(ge=0.0)
    estimated_liquidation_days: float = Field(ge=0.0)


class TurnoverMetrics(BaseModel):
    """Gross traded notional relative to average portfolio value."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    traded_notional: float = Field(ge=0.0)
    average_portfolio_value: float = Field(gt=0.0)
    turnover_ratio: float = Field(ge=0.0)
