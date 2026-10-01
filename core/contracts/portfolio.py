from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Position(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_symbol: str = Field(min_length=1)
    quantity: float
    average_price: float = Field(gt=0)
    market_price: float = Field(gt=0)

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price


class Portfolio(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    portfolio_id: str = Field(min_length=1)
    timestamp: datetime
    base_currency: str = Field(min_length=3, max_length=3)
    cash: float = Field(ge=0)
    positions: tuple[Position, ...] = ()

    @property
    def gross_market_value(self) -> float:
        return sum(abs(position.market_value) for position in self.positions)

    @property
    def net_market_value(self) -> float:
        return sum(position.market_value for position in self.positions)


class TargetAllocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_symbol: str = Field(min_length=1)
    target_weight: float = Field(ge=0, le=1)


class SizingResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_symbol: str = Field(min_length=1)
    target_weight: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
