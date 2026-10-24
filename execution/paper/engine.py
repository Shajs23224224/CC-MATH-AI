"""Deterministic in-memory paper trading; no live broker or exchange calls."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from math import isfinite
from typing import Any, Literal, cast, Literal, cast

from core.contracts import RiskAction
from execution.paper.contracts import (
    AuditEvent,
    AuditEventType,
    ExecutionQuote,
    PaperAccountSnapshot,
    PaperFill,
    PaperOrder,
    PaperOrderRequest,
    PaperPosition,
)
from execution.policies.paper import PaperExecutionPolicy
from risk.governor.contracts import RiskGovernorResult

_ZERO_HASH = "0" * 64


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PaperTradingEngine:
    """Thread-safe paper simulator with risk gating, idempotency and hash-chain audit.

    The engine is intentionally in-memory and long-only. It does not implement a live
    adapter, credential storage, shorting, partial fills, queue priority or persistence.
    """

    def __init__(
        self,
        account_id: str,
        initial_cash: float,
        policy: PaperExecutionPolicy | None = None,
    ) -> None:
        if not account_id.strip():
            raise ValueError("account_id must not be empty")
        if not isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        self.account_id = account_id
        self.initial_cash = float(initial_cash)
        self.policy = policy or PaperExecutionPolicy()
        self._cash = float(initial_cash)
        self._orders: dict[str, PaperOrder] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._fills: list[PaperFill] = []
        self._positions: dict[str, PaperPosition] = {}
        self._audit: list[AuditEvent] = []
        self._quotes: dict[str, ExecutionQuote] = {}
        self._commissions_paid = 0.0
        self._last_timestamp: datetime | None = None
        self._lock = threading.RLock()

    @property
    def orders(self) -> tuple[PaperOrder, ...]:
        with self._lock:
            return tuple(self._orders[key] for key in sorted(self._orders))

    @property
    def fills(self) -> tuple[PaperFill, ...]:
        with self._lock:
            return tuple(self._fills)

    @property
    def audit_events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._audit)

    def submit_order(
        self,
        request: PaperOrderRequest,
        risk_decision: RiskGovernorResult,
    ) -> PaperOrder:
        """Accept or reject a request only against an explicit F38 decision.

        Repeating the same idempotency key and exact payload returns the original order.
        Reusing that key for a different payload raises instead of duplicating exposure.
        """
        with self._lock:
            request_json = _canonical(request.model_dump(mode="json"))
            previous = self._idempotency.get(request.idempotency_key)
            if previous is not None:
                order_id, prior_request_json = previous
                if prior_request_json != request_json:
                    raise ValueError("idempotency_key was already used for a different request")
                return self._orders[order_id]

            if self._last_timestamp is not None and request.submitted_at < self._last_timestamp:
                raise ValueError("order timestamp must be monotonic")
            order_id = _sha256(f"{self.account_id}|{request.idempotency_key}")
            risk_action = cast(Literal["ALLOW", "REDUCE", "BLOCK"], risk_decision.action.value)
            approved_position = next(
                (
                    item
                    for item in risk_decision.positions
                    if item.asset_symbol == request.asset_symbol
                ),
                None,
            )
            reason: str | None = None
            risk_age = (request.submitted_at - risk_decision.as_of).total_seconds()
            approved_quantity = (
                0.0 if approved_position is None else approved_position.approved_quantity
            )
            if risk_age < 0:
                reason = "F38 risk decision timestamp is later than order submission"
            elif risk_age > self.policy.max_risk_decision_age_seconds:
                reason = "F38 risk decision is stale under the paper execution policy"
            elif risk_decision.action == RiskAction.BLOCK:
                reason = "F38 risk governor BLOCK vetoed order submission"
            elif approved_position is None:
                reason = "asset is absent from the F38 approved-position set"
            elif approved_position.status == "BLOCKED" or approved_quantity <= 0.0:
                reason = "F38 approved quantity is zero or position is blocked"
            elif request.quantity > approved_quantity + 1e-9:
                reason = "requested quantity exceeds F38 approved quantity"
            elif request.order_type == "MARKET" and not self.policy.allow_market_orders:
                reason = "market orders are disabled by the paper execution policy"
            elif request.order_type == "LIMIT" and not self.policy.allow_limit_orders:
                reason = "limit orders are disabled by the paper execution policy"
            elif request.side == "BUY":
                held = self._positions.get(request.asset_symbol)
                if held is None:
                    held = PaperPosition(asset_symbol=request.asset_symbol, quantity=0.0)
                pending_buys = sum(
                    order.request.quantity
                    for order in self._orders.values()
                    if order.status == "ACCEPTED"
                    and order.request.asset_symbol == request.asset_symbol
                    and order.request.side == "BUY"
                )
                if held.quantity + pending_buys + request.quantity > approved_quantity + 1e-9:
                    reason = "order would exceed F38 approved target after pending buys"
            else:
                held = self._positions.get(request.asset_symbol)
                pending_sells = sum(
                    order.request.quantity
                    for order in self._orders.values()
                    if order.status == "ACCEPTED"
                    and order.request.asset_symbol == request.asset_symbol
                    and order.request.side == "SELL"
                )
                available = 0.0 if held is None else held.quantity
                if request.quantity > available - pending_sells + 1e-9:
                    reason = "sell quantity exceeds unreserved paper holdings"

            status: Literal["REJECTED", "ACCEPTED"] = "REJECTED" if reason else "ACCEPTED"
            updated_at = request.submitted_at
            order = PaperOrder(
                order_id=order_id,
                request=request,
                status=status,
                risk_action=risk_action,
                risk_governor_fingerprint=risk_decision.governor_fingerprint,
                approved_quantity=approved_quantity,
                rejection_reason=reason,
                updated_at=updated_at,
            )
            self._orders[order_id] = order
            self._idempotency[request.idempotency_key] = (order_id, request_json)
            self._last_timestamp = request.submitted_at
            self._append_audit(
                "ORDER_REJECTED" if reason else "ORDER_ACCEPTED",
                updated_at,
                order_id,
                {
                    "request": request.model_dump(mode="json"),
                    "risk_action": risk_action,
                    "risk_governor_fingerprint": risk_decision.governor_fingerprint,
                    "reason": reason,
                    "policy_id": self.policy.policy_id,
                    "policy_version": self.policy.version,
                },
            )
            return order

    def process_quote(
        self,
        quote: ExecutionQuote,
        *,
        as_of: datetime | None = None,
    ) -> tuple[PaperFill, ...]:
        """Mark an asset and match eligible orders against one point-in-time quote."""
        with self._lock:
            evaluated_at = as_of or quote.timestamp
            if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
                raise ValueError("as_of must be timezone-aware")
            age = (evaluated_at - quote.timestamp).total_seconds()
            if age < 0:
                raise ValueError("quote timestamp cannot be later than evaluation time")
            if age > self.policy.max_quote_age_seconds:
                raise ValueError("quote is stale under the configured paper policy")
            previous_quote = self._quotes.get(quote.asset_symbol)
            if previous_quote is not None and quote.timestamp < previous_quote.timestamp:
                raise ValueError("out-of-order quote rejected")
            if self._last_timestamp is not None and evaluated_at < self._last_timestamp:
                raise ValueError("paper account event time must be monotonic")

            self._quotes[quote.asset_symbol] = quote
            self._last_timestamp = evaluated_at
            position = self._positions.get(quote.asset_symbol)
            if position is not None:
                mark = (quote.bid + quote.ask) / 2.0
                self._positions[quote.asset_symbol] = position.model_copy(
                    update={
                        "last_mark": mark,
                        "unrealized_pnl": (mark - position.average_cost) * position.quantity,
                        "market_value": mark * position.quantity,
                    }
                )

            new_fills: list[PaperFill] = []
            for order in tuple(self._orders.values()):
                if order.status != "ACCEPTED" or order.request.asset_symbol != quote.asset_symbol:
                    continue
                fill_price = self._eligible_fill_price(order.request, quote)
                if fill_price is None:
                    continue
                fill_or_reason = self._execute(order, quote, fill_price, evaluated_at)
                if isinstance(fill_or_reason, PaperFill):
                    new_fills.append(fill_or_reason)
            return tuple(new_fills)

    def cancel_order(self, order_id: str, *, at: datetime) -> PaperOrder:
        """Cancel an unfilled accepted order; filled/rejected/cancelled orders are terminal."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                raise KeyError("unknown paper order")
            if at.tzinfo is None or at.utcoffset() is None:
                raise ValueError("cancellation timestamp must be timezone-aware")
            if self._last_timestamp is not None and at < self._last_timestamp:
                raise ValueError("paper account event time must be monotonic")
            if order.status != "ACCEPTED":
                raise ValueError(f"cannot cancel an order in {order.status} state")
            updated = order.model_copy(update={"status": "CANCELLED", "updated_at": at})
            self._orders[order_id] = updated
            self._last_timestamp = at
            self._append_audit("ORDER_CANCELLED", at, order_id, {"reason": "cancel requested"})
            return updated

    def snapshot(self, *, at: datetime | None = None) -> PaperAccountSnapshot:
        """Return a deterministic mark-to-market account snapshot."""
        with self._lock:
            timestamp = at or self._last_timestamp
            if timestamp is None:
                raise ValueError("account snapshot requires at or at least one processed quote")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("snapshot timestamp must be timezone-aware")
            positions = tuple(self._positions[key] for key in sorted(self._positions))
            market_value = sum(position.quantity * position.last_mark for position in positions)
            unrealized = sum(position.unrealized_pnl for position in positions)
            realized = sum(position.realized_pnl for position in positions)
            return PaperAccountSnapshot(
                account_id=self.account_id,
                base_currency=self.policy.base_currency,
                timestamp=timestamp,
                initial_cash=self.initial_cash,
                cash=self._cash,
                market_value=market_value,
                equity=self._cash + market_value,
                realized_pnl=realized,
                unrealized_pnl=unrealized,
                commissions_paid=self._commissions_paid,
                positions=positions,
            )

    def verify_audit_chain(self) -> bool:
        """Verify sequence, previous hashes and every event hash from the genesis hash."""
        with self._lock:
            previous_hash = _ZERO_HASH
            for index, event in enumerate(self._audit, start=1):
                if event.sequence != index or event.previous_hash != previous_hash:
                    return False
                material = _canonical(
                    {
                        "sequence": event.sequence,
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "entity_id": event.entity_id,
                        "payload_json": event.payload_json,
                        "previous_hash": previous_hash,
                    }
                )
                expected = _sha256(material)
                if expected != event.event_hash:
                    return False
                previous_hash = event.event_hash
            return True

    def _eligible_fill_price(
        self,
        request: PaperOrderRequest,
        quote: ExecutionQuote,
    ) -> float | None:
        if request.order_type == "MARKET":
            reference = quote.ask if request.side == "BUY" else quote.bid
            direction = 1.0 if request.side == "BUY" else -1.0
            price = reference * (1.0 + direction * self.policy.market_slippage_bps / 10_000.0)
            if not isfinite(price) or price <= 0:
                raise ArithmeticError("paper execution produced an invalid fill price")
            return price
        assert request.limit_price is not None
        if request.side == "BUY" and quote.ask <= request.limit_price:
            return min(quote.ask, request.limit_price)
        if request.side == "SELL" and quote.bid >= request.limit_price:
            return max(quote.bid, request.limit_price)
        return None

    def _execute(
        self,
        order: PaperOrder,
        quote: ExecutionQuote,
        fill_price: float,
        at: datetime,
    ) -> PaperFill | None:
        request = order.request
        quantity = request.quantity
        notional = quantity * fill_price
        commission = notional * self.policy.commission_bps / 10_000.0
        if request.side == "BUY":
            if notional + commission > self._cash + 1e-9:
                self._reject_accepted_order(order, at, "insufficient paper cash for full fill")
                return None
            current = self._positions.get(request.asset_symbol)
            old_quantity = 0.0 if current is None else current.quantity
            old_cost = 0.0 if current is None else current.average_cost
            new_quantity = old_quantity + quantity
            average_cost = (old_quantity * old_cost + notional + commission) / new_quantity
            self._cash -= notional + commission
            self._positions[request.asset_symbol] = PaperPosition(
                asset_symbol=request.asset_symbol,
                quantity=new_quantity,
                average_cost=average_cost,
                last_mark=(quote.bid + quote.ask) / 2.0,
                realized_pnl=0.0 if current is None else current.realized_pnl,
                unrealized_pnl=(quote.bid + quote.ask) / 2.0 * new_quantity
                - average_cost * new_quantity,
                market_value=(quote.bid + quote.ask) / 2.0 * new_quantity,
            )
        else:
            current = self._positions.get(request.asset_symbol)
            if current is None or quantity > current.quantity + 1e-9:
                self._reject_accepted_order(
                    order, at, "insufficient paper holdings for full sell fill"
                )
                return None
            remaining = max(0.0, current.quantity - quantity)
            trade_pnl = (fill_price - current.average_cost) * quantity
            realized = current.realized_pnl + trade_pnl - commission
            self._cash += notional - commission
            mark = (quote.bid + quote.ask) / 2.0
            self._positions[request.asset_symbol] = PaperPosition(
                asset_symbol=request.asset_symbol,
                quantity=remaining,
                average_cost=current.average_cost if remaining else 0.0,
                last_mark=mark,
                realized_pnl=realized,
                unrealized_pnl=(mark - current.average_cost) * remaining,
                market_value=mark * remaining,
            )
        reference_price = quote.ask if request.side == "BUY" else quote.bid
        slip_cost = quantity * abs(fill_price - reference_price)
        fill_id = _sha256(f"{order.order_id}|fill|1")
        fill = PaperFill(
            fill_id=fill_id,
            order_id=order.order_id,
            asset_symbol=request.asset_symbol,
            side=request.side,
            quantity=quantity,
            reference_price=reference_price,
            fill_price=fill_price,
            notional=notional,
            commission=commission,
            slippage_cost=slip_cost,
            timestamp=at,
        )
        self._fills.append(fill)
        self._commissions_paid += commission
        self._orders[order.order_id] = order.model_copy(
            update={
                "status": "FILLED",
                "filled_quantity": quantity,
                "average_fill_price": fill_price,
                "commission_paid": commission,
                "updated_at": at,
            }
        )
        self._last_timestamp = at
        self._append_audit(
            "ORDER_FILLED",
            at,
            order.order_id,
            {
                "fill": fill.model_dump(mode="json"),
                "cash_after": self._cash,
                "position_quantity_after": self._positions[request.asset_symbol].quantity,
            },
        )
        return fill

    def _reject_accepted_order(self, order: PaperOrder, at: datetime, reason: str) -> None:
        self._orders[order.order_id] = order.model_copy(
            update={"status": "REJECTED", "rejection_reason": reason, "updated_at": at}
        )
        self._append_audit("ORDER_REJECTED", at, order.order_id, {"reason": reason})

    def _append_audit(
        self,
        event_type: AuditEventType,
        timestamp: datetime,
        entity_id: str,
        payload: dict[str, Any],
    ) -> None:
        previous_hash = self._audit[-1].event_hash if self._audit else _ZERO_HASH
        payload_json = _canonical(payload)
        sequence = len(self._audit) + 1
        material = _canonical(
            {
                "sequence": sequence,
                "event_type": event_type,
                "timestamp": timestamp.isoformat(),
                "entity_id": entity_id,
                "payload_json": payload_json,
                "previous_hash": previous_hash,
            }
        )
        self._audit.append(
            AuditEvent(
                sequence=sequence,
                event_type=event_type,
                timestamp=timestamp,
                entity_id=entity_id,
                payload_json=payload_json,
                previous_hash=previous_hash,
                event_hash=_sha256(material),
            )
        )
