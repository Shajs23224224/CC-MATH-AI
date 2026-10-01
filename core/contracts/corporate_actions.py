from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import CorporateActionType
from .market import Asset


class CorporateAction(BaseModel):
    """A dated corporate event with explicit point-in-time provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    action_type: CorporateActionType
    announced_at: date | None = None
    ex_date: date | None = None
    record_date: date | None = None
    payable_date: date | None = None

    ratio_numerator: float | None = Field(default=None, gt=0)
    ratio_denominator: float | None = Field(default=None, gt=0)

    cash_amount: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, min_length=3, max_length=3)

    new_symbol: str | None = None
    related_symbol: str | None = None

    source: str = Field(min_length=1)
    source_event_id: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> CorporateAction:
        if self.action_type in {
            CorporateActionType.SPLIT,
            CorporateActionType.REVERSE_SPLIT,
        }:
            if self.ex_date is None:
                raise ValueError("split actions require ex_date")
            if self.ratio_numerator is None or self.ratio_denominator is None:
                raise ValueError("split actions require ratio_numerator and ratio_denominator")

        if self.action_type in {
            CorporateActionType.DIVIDEND,
            CorporateActionType.SPECIAL_DIVIDEND,
        }:
            if self.ex_date is None:
                raise ValueError("dividend actions require ex_date")
            if self.cash_amount is None:
                raise ValueError("dividend actions require cash_amount")

        if self.action_type == CorporateActionType.TICKER_CHANGE:
            if self.ex_date is None and self.announced_at is None:
                raise ValueError("ticker changes require an announcement or effective date")
            if not self.new_symbol:
                raise ValueError("ticker changes require new_symbol")

        if self.announced_at and self.ex_date and self.ex_date < self.announced_at:
            raise ValueError("ex_date cannot precede announced_at")

        if self.record_date and self.ex_date and self.record_date < self.ex_date:
            raise ValueError("record_date cannot precede ex_date")

        if self.payable_date and self.record_date and self.payable_date < self.record_date:
            raise ValueError("payable_date cannot precede record_date")

        return self

    @property
    def split_factor(self) -> float:
        if self.ratio_numerator is None or self.ratio_denominator is None:
            raise ValueError("action does not contain a split ratio")
        return self.ratio_numerator / self.ratio_denominator


class CorporateActionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    as_of: date | None = None
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def validate_window(self) -> CorporateActionRequest:
        if self.start and self.end and self.end < self.start:
            raise ValueError("end must be on or after start")
        return self


class CorporateActionBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: CorporateActionRequest
    actions_count: int = Field(ge=0)
