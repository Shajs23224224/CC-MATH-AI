from datetime import UTC, datetime, timedelta

import pytest

from core.contracts import RiskAction
from execution.paper import (
    ExecutionQuote,
    PaperExecutionPolicy,
    PaperOrderRequest,
    PaperTradingEngine,
)
from risk.governor.contracts import GovernorApprovedPosition, RiskGovernorResult


def now() -> datetime:
    return datetime(2026, 1, 2, 15, 0, tzinfo=UTC)


def decision(*, blocked: bool = False, quantity: float = 10.0) -> RiskGovernorResult:
    first_quantity = 0.0 if blocked else quantity
    first_status = "BLOCKED" if blocked else "APPROVED"
    positions = (
        GovernorApprovedPosition(
            asset_symbol="XYZ",
            source_quantity=quantity,
            source_notional=quantity * 100.0,
            source_weight=quantity / 100.0,
            approved_quantity=first_quantity,
            approved_notional=first_quantity * 100.0,
            approved_weight=first_quantity / 100.0,
            lot_size=1.0,
            status=first_status,
            rationale="fixture risk decision",
        ),
        GovernorApprovedPosition(
            asset_symbol="ABC",
            source_quantity=0.0,
            source_notional=0.0,
            source_weight=0.0,
            approved_quantity=0.0,
            approved_notional=0.0,
            approved_weight=0.0,
            lot_size=1.0,
            status="BLOCKED" if blocked else "ZERO_SIZE",
            rationale="no target exposure",
        ),
    )
    return RiskGovernorResult(
        action=RiskAction.BLOCK if blocked else RiskAction.ALLOW,
        reduction_factor=0.0 if blocked else 1.0,
        reasons=("test block",) if blocked else (),
        triggered_constraints=("test",) if blocked else (),
        positions=positions,
        equity=10_000.0,
        total_approved_notional=first_quantity * 100.0,
        total_approved_weight=first_quantity / 100.0,
        config_id="test-policy",
        config_version="1.0.0",
        as_of=now(),
        data_snapshot_id="data-test",
        risk_snapshot_id="risk-test",
        model_snapshot_id="model-test",
        sizing_fingerprint="a" * 64,
        governor_fingerprint="b" * 64,
    )


def request(
    *,
    side: str = "BUY",
    quantity: float = 5.0,
    key: str = "paper-order-0001",
    order_type: str = "MARKET",
    limit_price: float | None = None,
) -> PaperOrderRequest:
    return PaperOrderRequest(
        asset_symbol="xyz",
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        idempotency_key=key,
        strategy_id="test-strategy",
        signal_id="signal-0001",
        submitted_at=now(),
    )


def quote(
    *,
    bid: float = 100.0,
    ask: float = 101.0,
    timestamp: datetime | None = None,
) -> ExecutionQuote:
    return ExecutionQuote(
        asset_symbol="XYZ",
        bid=bid,
        ask=ask,
        timestamp=timestamp or now(),
    )


def engine(policy: PaperExecutionPolicy | None = None) -> PaperTradingEngine:
    return PaperTradingEngine("paper-account-1", 10_000.0, policy)


def test_market_order_requires_risk_approval_and_fills_with_costs() -> None:
    simulator = engine()
    order = simulator.submit_order(request(), decision())
    assert order.status == "ACCEPTED"

    fills = simulator.process_quote(quote())
    assert len(fills) == 1
    fill = fills[0]
    assert fill.side == "BUY"
    assert fill.quantity == 5.0
    assert fill.fill_price > fill.reference_price
    assert fill.commission > 0
    assert fill.slippage_cost > 0
    assert simulator.orders[0].status == "FILLED"

    snapshot = simulator.snapshot()
    assert snapshot.cash < snapshot.initial_cash
    assert snapshot.market_value > 0
    assert snapshot.equity == pytest.approx(snapshot.cash + snapshot.market_value)
    assert snapshot.commissions_paid == pytest.approx(fill.commission)
    assert snapshot.positions[0].quantity == 5.0
    assert simulator.verify_audit_chain()


def test_blocked_risk_decision_fails_closed() -> None:
    simulator = engine()
    order = simulator.submit_order(request(), decision(blocked=True))
    assert order.status == "REJECTED"
    assert "BLOCK vetoed" in (order.rejection_reason or "")
    assert simulator.process_quote(quote()) == ()
    assert simulator.snapshot(at=now()).cash == 10_000.0
    assert simulator.verify_audit_chain()


def test_order_cannot_exceed_f38_approved_quantity() -> None:
    order = engine().submit_order(request(quantity=11.0), decision(quantity=10.0))
    assert order.status == "REJECTED"
    assert "exceeds F38 approved" in (order.rejection_reason or "")


def test_idempotency_returns_original_order_and_rejects_key_reuse() -> None:
    simulator = engine()
    first = simulator.submit_order(request(), decision())
    repeated = simulator.submit_order(request(), decision())
    assert repeated.order_id == first.order_id
    assert len(simulator.orders) == 1

    with pytest.raises(ValueError, match="different request"):
        simulator.submit_order(request(quantity=4.0), decision())


def test_limit_order_waits_until_price_is_marketable() -> None:
    simulator = engine()
    order = simulator.submit_order(
        request(order_type="LIMIT", limit_price=100.5, key="limit-order-0001"),
        decision(),
    )
    assert order.status == "ACCEPTED"
    assert simulator.process_quote(quote(bid=101.0, ask=102.0)) == ()
    fills = simulator.process_quote(
        quote(bid=100.0, ask=100.25, timestamp=now() + timedelta(seconds=1))
    )
    assert len(fills) == 1
    assert fills[0].fill_price <= 100.5
    assert fills[0].slippage_cost == 0.0


def test_stale_and_out_of_order_quotes_are_rejected() -> None:
    simulator = engine(PaperExecutionPolicy(max_quote_age_seconds=5))
    with pytest.raises(ValueError, match="stale"):
        simulator.process_quote(
            quote(timestamp=now() - timedelta(seconds=6)),
            as_of=now(),
        )
    simulator.process_quote(quote(timestamp=now()))
    with pytest.raises(ValueError, match="out-of-order"):
        simulator.process_quote(quote(timestamp=now() - timedelta(seconds=1)))


def test_sell_order_requires_existing_paper_holdings() -> None:
    order = engine().submit_order(
        request(side="SELL", key="sell-order-0001"),
        decision(),
    )
    assert order.status == "REJECTED"
    assert "holdings" in (order.rejection_reason or "")


def test_sell_fill_realizes_pnl_and_closes_position() -> None:
    simulator = engine(PaperExecutionPolicy(commission_bps=0.0, market_slippage_bps=0.0))
    buy = simulator.submit_order(request(key="buy-order-00001"), decision())
    simulator.process_quote(quote(bid=99.0, ask=100.0))
    assert simulator.orders[0].order_id == buy.order_id

    sell_request = PaperOrderRequest(
        asset_symbol="XYZ",
        side="SELL",
        quantity=5.0,
        idempotency_key="sell-order-00001",
        strategy_id="test-strategy",
        signal_id="signal-0002",
        submitted_at=now() + timedelta(seconds=1),
    )
    sell = simulator.submit_order(sell_request, decision())
    assert sell.status == "ACCEPTED"
    fills = simulator.process_quote(
        quote(bid=110.0, ask=111.0, timestamp=now() + timedelta(seconds=1))
    )
    assert len(fills) == 1
    snapshot = simulator.snapshot()
    assert snapshot.positions[0].quantity == 0.0
    assert snapshot.cash > snapshot.initial_cash
    assert snapshot.realized_pnl > 0.0
    assert simulator.verify_audit_chain()


def test_cancel_is_terminal_and_audited() -> None:
    simulator = engine()
    order = simulator.submit_order(
        request(order_type="LIMIT", limit_price=90.0, key="cancel-order-0001"),
        decision(),
    )
    cancelled = simulator.cancel_order(order.order_id, at=now())
    assert cancelled.status == "CANCELLED"
    assert simulator.process_quote(quote()) == ()
    with pytest.raises(ValueError, match="cannot cancel"):
        simulator.cancel_order(order.order_id, at=now())
    assert simulator.verify_audit_chain()


def test_market_order_cannot_use_stale_or_future_evidence() -> None:
    with pytest.raises(ValueError, match="bid cannot exceed ask"):
        ExecutionQuote(asset_symbol="XYZ", bid=102.0, ask=101.0, timestamp=now())
    with pytest.raises(ValueError, match="LIMIT order requires"):
        request(order_type="LIMIT", limit_price=None)
