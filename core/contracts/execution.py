from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import OrderSide, OrderType


class Order(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(min_length=1)
    asset_symbol: str = Field(min_length=1)
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0)
    limit_price: float | None = Field(default=None, gt=0)
    created_at: datetime


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime
    commission: float = Field(ge=0)


class ExecutionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    fills: tuple[Fill, ...] = ()
