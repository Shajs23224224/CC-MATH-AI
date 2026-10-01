from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import SignalType


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    direction: int = Field(ge=-1, le=1)
    contribution: float
    rationale: str = Field(min_length=1)


class Forecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: str = Field(min_length=1)
    expected_return: float
    expected_risk: float = Field(ge=0)
    uncertainty: float = Field(ge=0)


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_symbol: str = Field(min_length=1)
    timestamp: datetime
    signal: SignalType
    probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    forecast: Forecast
    evidence: tuple[Evidence, ...] = ()


class DecisionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(min_length=1)
    asset_symbol: str = Field(min_length=1)
    timestamp: datetime
    signal: SignalType
    signal_probability: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)
    expected_return: float
    expected_risk: float = Field(ge=0)
    risk_adjusted_score: float
    recommended_horizon: str = Field(min_length=1)
    suggested_position_size: float = Field(ge=0)
    regime: str = Field(min_length=1)
    data_snapshot_id: str = Field(min_length=1)
    model_versions: tuple[str, ...] = ()
    feature_versions: tuple[str, ...] = ()
