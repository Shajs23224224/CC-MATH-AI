"""Sum-of-the-parts valuation."""

from __future__ import annotations

import math

from core.contracts import ValuationAssumptions, ValuationResult


def sotp_equity_value(
    segment_values: tuple[float, ...],
    *,
    segment_basis: str,
    debt: float = 0.0,
    cash: float = 0.0,
) -> float:
    if not segment_values:
        raise ValueError("at least one segment value is required")
    if any(not math.isfinite(value) or value < 0.0 for value in segment_values):
        raise ValueError("segment values must be finite and non-negative")
    debt = _non_negative(debt, "debt")
    cash = _non_negative(cash, "cash")

    total = math.fsum(segment_values)
    if segment_basis == "enterprise":
        return total - debt + cash
    if segment_basis == "equity":
        if debt != 0.0 or cash != 0.0:
            raise ValueError("debt and cash must be zero when segments are equity values")
        return total
    raise ValueError("segment_basis must be 'enterprise' or 'equity'")


def sotp_result(
    segment_values: tuple[float, ...],
    *,
    segment_basis: str,
    shares_outstanding: float,
    assumptions: ValuationAssumptions,
    debt: float = 0.0,
    cash: float = 0.0,
) -> ValuationResult:
    equity_value = sotp_equity_value(
        segment_values,
        segment_basis=segment_basis,
        debt=debt,
        cash=cash,
    )
    shares = _positive(shares_outstanding, "shares_outstanding")
    per_share = equity_value / shares
    return ValuationResult(
        method="sotp",
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        basis="equity",
        unit="currency",
        value=equity_value,
        equity_value=equity_value,
        per_share_value=per_share,
    )


def _non_negative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value
