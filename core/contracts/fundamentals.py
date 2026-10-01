from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .market import Asset


class FundamentalSnapshot(BaseModel):
    """Point-in-time fundamental facts.

    period_end identifies the accounting period; reported_at identifies when
    the information became available to the system. They must not be conflated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    period_end: date
    reported_at: date
    revenue: float | None = None
    ebitda: float | None = None
    ebit: float | None = None
    net_income: float | None = None
    eps: float | None = None
    free_cash_flow: float | None = None
    capex: float | None = None
    cash: float | None = None
    debt: float | None = None
    equity: float | None = None
    dividends: float | None = None
    shares_outstanding: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_reporting_dates(self) -> FundamentalSnapshot:
        if self.reported_at < self.period_end:
            raise ValueError("reported_at cannot precede period_end")
        return self
