from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import Frequency
from .market import Asset


class MarketDataRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    frequency: Frequency = Frequency.DAILY
    start: datetime | None = None
    end: datetime | None = None
    adjusted: bool = False

    @model_validator(mode="after")
    def validate_window(self) -> MarketDataRequest:
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class MarketDataBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: MarketDataRequest
    bars_count: int = Field(ge=0)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
