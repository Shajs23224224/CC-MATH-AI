"""Deterministic F37 position sizing with portfolio and liquidity caps."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from portfolio.sizing.contracts import (
    PositionSizingConfig,
    PositionSizingRequest,
    SizedPosition,
    SizingResult,
)


def _fingerprint(payload: Any) -> str:
    """Hash a canonical JSON representation for deterministic replay."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _round_down_to_lot(quantity: float, lot_size: float) -> float:
    """Round long-only shares/units down, never upward through a risk limit."""
    if quantity <= 0.0:
        return 0.0
    lots = math.floor(quantity / lot_size + 1e-12)
    rounded = lots * lot_size
    return 0.0 if rounded <= 0.0 else rounded


def size_positions(
    request: PositionSizingRequest,
    config: PositionSizingConfig | None = None,
) -> SizingResult:
    """Convert valid F36 target weights into bounded, auditable share quantities.

    F36 target weights are hard upper bounds. Sizing may reduce exposure, leave cash
    unallocated, or return zero; it may never increase the approved target or leverage.
    """
    policy = config or PositionSizingConfig()
    optimization = request.portfolio_optimization
    source_weights = {
        allocation.asset_symbol: allocation.target_weight
        for allocation in optimization.allocations
    }
    total_source_weight = math.fsum(source_weights.values())
    raw: list[dict[str, Any]] = []

    for asset in request.assets:
        source_weight = source_weights[asset.asset_symbol]
        max_weight = min(source_weight, policy.maximum_position_weight)
        weight_limited = source_weight > policy.maximum_position_weight + 1e-12
        reasons: list[str] = []
        risk_budget = 0.0
        stop_distance = (
            asset.price - asset.stop_loss_price
            if asset.stop_loss_price is not None
            else None
        )

        if source_weight <= 0.0:
            desired_weight = 0.0
            reasons.append("F36 target weight is zero")
        elif policy.method == "fixed_fractional":
            desired_weight = min(max_weight, policy.fixed_fractional_weight)
            reasons.append("fixed fraction of equity, capped by F36 allocation and position limit")
        elif policy.method == "volatility_based":
            if asset.volatility <= 0.0:
                desired_weight = 0.0
                reasons.append("zero or missing volatility cannot support volatility-based sizing")
            else:
                desired_weight = min(
                    max_weight, policy.volatility_budget_fraction / asset.volatility
                )
                reasons.append("horizon volatility budget divided by asset volatility")
        elif policy.method == "risk_based":
            if stop_distance is None:
                desired_weight = 0.0
                reasons.append("risk-based sizing requires an explicit stop-loss price")
            elif total_source_weight <= 0.0:
                desired_weight = 0.0
                reasons.append("F36 provides no positive target weight for risk budgeting")
            else:
                risk_budget = (
                    request.equity
                    * policy.risk_budget_fraction
                    * source_weight
                    / total_source_weight
                )
                ideal_quantity = risk_budget / stop_distance
                desired_weight = min(max_weight, ideal_quantity * asset.price / request.equity)
                reasons.append("pro-rata equity risk budget divided by loss per unit at stop")
        elif policy.method in {"kelly", "fractional_kelly"}:
            if asset.volatility <= 0.0:
                desired_weight = 0.0
                reasons.append("Kelly sizing requires positive horizon volatility")
            else:
                excess_return = asset.expected_return - policy.risk_free_rate
                full_kelly = max(0.0, excess_return / (asset.volatility * asset.volatility))
                multiplier = (
                    policy.fractional_kelly_multiplier
                    if policy.method == "fractional_kelly"
                    else 1.0
                )
                desired_weight = min(max_weight, full_kelly * multiplier)
                reasons.append(
                    "Kelly excess-return/variance estimate"
                    + (
                        " scaled by fractional Kelly multiplier"
                        if policy.method == "fractional_kelly"
                        else ""
                    )
                )
        else:
            desired_weight = min(max_weight, source_weight * asset.confidence)
            reasons.append("F36 allocation scaled by external confidence in [0, 1]")

        desired_weight = max(0.0, min(desired_weight, max_weight))
        desired_quantity = desired_weight * request.equity / asset.price
        liquidity_cap_quantity = _round_down_to_lot(
            asset.average_daily_volume * policy.max_participation_rate,
            asset.lot_size,
        )
        quantity_before_liquidity = _round_down_to_lot(desired_quantity, asset.lot_size)
        liquidity_limited = quantity_before_liquidity > liquidity_cap_quantity
        quantity = min(quantity_before_liquidity, liquidity_cap_quantity)
        notional = quantity * asset.price

        if quantity > 0.0 and notional + 1e-9 < policy.minimum_trade_notional:
            quantity = 0.0
            notional = 0.0
            reasons.append("rounded trade notional is below the configured minimum")
        if liquidity_limited:
            reasons.append("limited by maximum average-daily-volume participation")
        if weight_limited:
            reasons.append("capped by maximum position weight")
        if quantity == 0.0 and not reasons:
            reasons.append("no positive target quantity remains after constraints")

        target_weight = notional / request.equity
        estimated_loss = (
            quantity * stop_distance
            if stop_distance is not None and stop_distance > 0.0
            else 0.0
        )
        adv_participation = quantity / asset.average_daily_volume
        status = "SIZED" if quantity > 0.0 else "ZERO_SIZE"
        raw.append(
            {
                "asset_symbol": asset.asset_symbol,
                "source_target_weight": source_weight,
                "target_weight": target_weight,
                "target_quantity": quantity,
                "target_notional": notional,
                "risk_budget_amount": risk_budget,
                "estimated_loss_at_stop": estimated_loss,
                "adv_participation": adv_participation,
                "liquidity_limited": liquidity_limited,
                "position_weight_limited": weight_limited,
                "status": status,
                "rationale": "; ".join(dict.fromkeys(reasons)),
            }
        )

    allocations = tuple(SizedPosition.model_validate(row) for row in raw)
    positive = sum(item.target_quantity > 0.0 for item in allocations)
    result_status = (
        "ZERO_SIZE" if positive == 0 else "SIZED" if positive == len(allocations) else "PARTIAL"
    )
    audit_input = {
        "request": request.model_dump(mode="json"),
        "config": policy.model_dump(mode="json"),
    }
    return SizingResult(
        status=result_status,
        method=policy.method,
        config_id=policy.config_id,
        config_version=policy.version,
        allocations=allocations,
        total_target_weight=math.fsum(item.target_weight for item in allocations),
        total_target_notional=math.fsum(item.target_notional for item in allocations),
        total_risk_budget_amount=math.fsum(item.risk_budget_amount for item in allocations),
        equity=request.equity,
        base_currency=request.base_currency,
        as_of=request.as_of,
        horizon=request.horizon,
        market_data_snapshot_id=request.market_data_snapshot_id,
        portfolio_optimization_fingerprint=optimization.optimization_fingerprint,
        sizing_fingerprint=_fingerprint(audit_input),
    )


__all__ = ["size_positions"]
