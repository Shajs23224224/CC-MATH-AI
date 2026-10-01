from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import RiskAction


class RiskMetric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    timestamp: str
    methodology: str = Field(min_length=1)


class RiskConstraint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    limit: float
    metric_name: str = Field(min_length=1)
    hard: bool = True


class RiskEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: RiskAction
    reasons: tuple[str, ...] = ()
    metrics: tuple[RiskMetric, ...] = ()
    triggered_constraints: tuple[str, ...] = ()
