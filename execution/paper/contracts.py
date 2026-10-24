"""Validated, immutable contracts for F42 paper trading."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]
OrderStatus = Literal["ACCEPTED", "FILLED", "CANCELLED", "REJECTED"]
AuditEventType = Literal["ORDER_ACCEPTED", "ORDER_REJECTED", "ORDER_FILLED", "ORDER_CANCELLED"]


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


class ExecutionQuote(BaseModel):
    """Point-in-time bid/ask quote consumed only by the paper simulator."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    bid: float = Field(gt=0.0)
    ask: float = Field(gt=0.0)
    timestamp: datetime

    @field_validator("asset_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("asset_symbol must not be empty")
        return symbol

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def validate_market(self) -> ExecutionQuote:
        if self.bid > self.ask:
            raise ValueError("bid cannot exceed ask")
        return self


class PaperOrderRequest(BaseModel):
    """Idempotent order intent; it cannot name or select a live broker."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    side: OrderSide
    quantity: float = Field(gt=0.0)
    order_type: OrderType = "MARKET"
    limit_price: float | None = Field(default=None, gt=0.0)
    idempotency_key: str = Field(min_length=8, max_length=200)
    strategy_id: str = Field(min_length=1, max_length=160)
    signal_id: str = Field(min_length=1, max_length=200)
    submitted_at: datetime

    @field_validator("asset_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("asset_symbol must not be empty")
        return symbol

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value: float) -> float:
        if not isfinite(value) or value <= 0:
            raise ValueError("quantity must be finite and positive")
        return value

    @field_validator("submitted_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def validate_limit(self) -> PaperOrderRequest:
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.order_type == "MARKET" and self.limit_price is not None:
            raise ValueError("MARKET order cannot set limit_price")
        if self.limit_price is not None and not isfinite(self.limit_price):
            raise ValueError("limit_price must be finite")
        return self


class PaperOrder(BaseModel):
    """Order lifecycle record, including the risk decision that authorized it."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    order_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: PaperOrderRequest
    status: OrderStatus
    risk_action: Literal["ALLOW", "REDUCE", "BLOCK"]
    risk_governor_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_quantity: float = Field(ge=0.0)
    filled_quantity: float = Field(default=0.0, ge=0.0)
    average_fill_price: float | None = Field(default=None, gt=0.0)
    commission_paid: float = Field(default=0.0, ge=0.0)
    rejection_reason: str | None = None
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> PaperOrder:
        if self.filled_quantity > self.request.quantity + 1e-9:
            raise ValueError("filled quantity cannot exceed requested quantity")
        if self.status == "FILLED" and not isfinite(self.filled_quantity):
            raise ValueError("filled quantity must be finite")
        if self.status == "REJECTED" and not self.rejection_reason:
            raise ValueError("rejected orders require a reason")
        if self.status != "REJECTED" and self.rejection_reason is not None:
            raise ValueError("only rejected orders may have a rejection reason")
        return self


class PaperFill(BaseModel):
    """Simulated fill with explicit market reference, friction and accounting impact."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    fill_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_symbol: str
    side: OrderSide
    quantity: float = Field(gt=0.0)
    reference_price: float = Field(gt=0.0)
    fill_price: float = Field(gt=0.0)
    notional: float = Field(gt=0.0)
    commission: float = Field(ge=0.0)
    slippage_cost: float = Field(ge=0.0)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)


class PaperPosition(BaseModel):
    """Long-only position snapshot maintained by the paper accounting ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str
    quantity: float = Field(default=0.0, ge=0.0)
    average_cost: float = Field(default=0.0, ge=0.0)
    last_mark: float = Field(default=0.0, ge=0.0)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    market_value: float = Field(default=0.0, ge=0.0)


class PaperAccountSnapshot(BaseModel):
    """Consistent account state at a single mark timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    account_id: str
    base_currency: str
    timestamp: datetime
    initial_cash: float = Field(gt=0.0)
    cash: float
    market_value: float = Field(ge=0.0)
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    commissions_paid: float = Field(ge=0.0)
    positions: tuple[PaperPosition, ...]

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def validate_accounting_identities(self) -> PaperAccountSnapshot:
        """Reject snapshots whose stated totals disagree with their position ledger."""
        expected_market_value = fsum(
            position.quantity * position.last_mark for position in self.positions
        )
        expected_realized = fsum(position.realized_pnl for position in self.positions)
        expected_unrealized = fsum(position.unrealized_pnl for position in self.positions)
        if not isclose(self.equity, self.cash + self.market_value, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("equity must equal cash plus market_value")
        if not isclose(self.market_value, expected_market_value, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("market_value must equal the sum of marked positions")
        if not isclose(self.realized_pnl, expected_realized, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("realized_pnl must equal the sum of realized position PnL")
        if not isclose(self.unrealized_pnl, expected_unrealized, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("unrealized_pnl must equal the sum of unrealized position PnL")
        return self


class AuditEvent(BaseModel):
    """Append-only hash-chained audit event; payload is canonical JSON-compatible data."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sequence: int = Field(ge=1)
    event_type: AuditEventType
    timestamp: datetime
    entity_id: str
    payload_json: str
    previous_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)
