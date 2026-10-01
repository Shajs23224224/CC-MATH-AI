"""Reverse-DCF inference of terminal growth."""

from __future__ import annotations

import math

from core.contracts import ValuationAssumptions


def implied_terminal_growth(
    target_enterprise_value: float,
    cash_flows: tuple[float, ...],
    discount_rate: float,
    *,
    lower_bound: float = -0.95,
    upper_bound: float | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    if not math.isfinite(target_enterprise_value) or target_enterprise_value <= 0.0:
        raise ValueError("target_enterprise_value must be finite and positive")
    if not cash_flows or any(not math.isfinite(value) for value in cash_flows):
        raise ValueError("cash_flows must be a non-empty finite sequence")
    if cash_flows[-1] <= 0.0:
        raise ValueError("last cash flow must be positive for reverse DCF")
    if not 0.0 < discount_rate < 1.0 or not math.isfinite(discount_rate):
        raise ValueError("discount_rate must be between 0 and 1")
    if upper_bound is None:
        upper_bound = discount_rate - 1e-9
    if not lower_bound < upper_bound < discount_rate:
        raise ValueError("invalid reverse DCF growth bounds")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    explicit_pv = sum(
        cash_flow / (1.0 + discount_rate) ** period
        for period, cash_flow in enumerate(cash_flows, start=1)
    )

    def enterprise_value(growth: float) -> float:
        terminal_cf = cash_flows[-1] * (1.0 + growth)
        terminal_value = terminal_cf / (discount_rate - growth)
        return explicit_pv + terminal_value / (1.0 + discount_rate) ** len(cash_flows)

    low_value = enterprise_value(lower_bound)
    high_value = enterprise_value(upper_bound)
    if target_enterprise_value < low_value or target_enterprise_value > high_value:
        raise ValueError("target enterprise value is outside the reverse DCF bounds")

    low = lower_bound
    high = upper_bound
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        value = enterprise_value(mid)
        if abs(value - target_enterprise_value) <= tolerance:
            return mid
        if value < target_enterprise_value:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def reverse_dcf_result(
    target_enterprise_value: float,
    cash_flows: tuple[float, ...],
    assumptions: ValuationAssumptions,
) -> float:
    discount_rate = assumptions.discount_rate
    if discount_rate is None:
        raise ValueError("discount_rate is required")
    return implied_terminal_growth(
        target_enterprise_value,
        cash_flows,
        discount_rate,
    )
