from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .market import Asset


class MacroFrequency(str):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    DAILY = "daily"
    WEEKLY = "weekly"


class MacroDataRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    geography: str = Field(min_length=1)
    as_of: date | None = None
    period_start: date | None = None
    period_end: date | None = None
    frequency: str = Field(min_length=1)
    asset: Asset | None = None

    @model_validator(mode="after")
    def validate_period(self) -> "MacroDataRequest":
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_end < self.period_start
        ):
            raise ValueError("period_end must be on or after period_start")
        return self


class MacroObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    geography: str = Field(min_length=1)
    period_end: date
    released_at: date
    value: float
    unit: str = Field(min_length=1)
    frequency: str = Field(min_length=1)
    source: str = Field(min_length=1)
    tenor: str | None = None
    asset: Asset | None = None

    @model_validator(mode="after")
    def validate_release_date(self) -> "MacroObservation":
        if self.released_at < self.period_end:
            raise ValueError("released_at cannot precede period_end")
        return self


class MacroDataBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: MacroDataRequest
    observations_count: int = Field(ge=0)
    first_period_end: date | None = None
    last_period_end: date | None = None
