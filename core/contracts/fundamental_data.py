from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .market import Asset


class FundamentalDataRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    period_start: date | None = None
    period_end: date | None = None
    as_of: date | None = None
    limit: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_window(self) -> FundamentalDataRequest:
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_end < self.period_start
        ):
            raise ValueError("period_end must be on or after period_start")
        return self


class FundamentalDataBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: FundamentalDataRequest
    snapshots_count: int = Field(ge=0)
    first_period_end: date | None = None
    last_period_end: date | None = None
