"""Net-asset-value valuation."""

from __future__ import annotations

import math

from core.contracts import ValuationAssumptions, ValuationResult


def nav_value(assets: float, liabilities: float) -> float:
    assets = _non_negative(assets, "assets")
    liabilities = _non_negative(liabilities, "liabilities")
    return assets - liabilities


def nav_result(
    assets: float,
    liabilities: float,
    shares_outstanding: float,
    assumptions: ValuationAssumptions,
) -> ValuationResult:
    equity_value = nav_value(assets, liabilities)
    shares = _positive(shares_outstanding, "shares_outstanding")
    per_share = equity_value / shares
    return ValuationResult(
        method="nav",
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
