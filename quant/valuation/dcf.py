"""Discounted-cash-flow valuation models."""

from __future__ import annotations

import math

from core.contracts import SensitivityMatrix, ValuationAssumptions, ValuationResult


def dcf_value(
    cash_flows: tuple[float, ...],
    discount_rate: float,
    terminal_growth_rate: float,
) -> float:
    if not cash_flows:
        raise ValueError("at least one forecast cash flow is required")
    if any(not math.isfinite(value) for value in cash_flows):
        raise ValueError("cash flows must be finite")
    _validate_rates(discount_rate, terminal_growth_rate)
    present_value = sum(
        cash_flow / (1.0 + discount_rate) ** period
        for period, cash_flow in enumerate(cash_flows, start=1)
    )
    terminal_cash_flow = cash_flows[-1] * (1.0 + terminal_growth_rate)
    terminal_value = terminal_cash_flow / (discount_rate - terminal_growth_rate)
    return present_value + terminal_value / (1.0 + discount_rate) ** len(cash_flows)


def dcf_result(
    cash_flows: tuple[float, ...],
    debt: float,
    cash: float,
    shares_outstanding: float,
    assumptions: ValuationAssumptions,
) -> ValuationResult:
    discount_rate = _required(assumptions.discount_rate, "discount_rate")
    growth_rate = _required(assumptions.terminal_growth_rate, "terminal_growth_rate")
    enterprise_value = dcf_value(cash_flows, discount_rate, growth_rate)
    equity_value = enterprise_value - _non_negative(debt, "debt") + _non_negative(cash, "cash")
    shares = _positive(shares_outstanding, "shares_outstanding")
    per_share = equity_value / shares
    return ValuationResult(
        method="dcf",
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        basis="enterprise",
        unit="currency",
        value=enterprise_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        per_share_value=per_share,
    )


def dcf_sensitivity(
    cash_flows: tuple[float, ...],
    discount_rates: tuple[float, ...],
    terminal_growth_rates: tuple[float, ...],
    assumptions: ValuationAssumptions,
) -> SensitivityMatrix:
    if not discount_rates or not terminal_growth_rates:
        raise ValueError("sensitivity axes cannot be empty")
    values = tuple(
        tuple(
            dcf_value(cash_flows, discount_rate, growth_rate)
            for growth_rate in terminal_growth_rates
        )
        for discount_rate in discount_rates
    )
    return SensitivityMatrix(
        method="dcf",
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        row_parameter="discount_rate",
        column_parameter="terminal_growth_rate",
        row_values=discount_rates,
        column_values=terminal_growth_rates,
        values=values,
    )


def _required(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _non_negative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_rates(discount_rate: float, growth_rate: float) -> None:
    if not math.isfinite(discount_rate) or not 0.0 < discount_rate < 1.0:
        raise ValueError("discount_rate must be between 0 and 1")
    if not math.isfinite(growth_rate) or growth_rate >= discount_rate:
        raise ValueError("terminal_growth_rate must be lower than discount_rate")
