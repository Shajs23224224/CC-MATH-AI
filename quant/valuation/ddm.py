"""Dividend-discount valuation models."""

from __future__ import annotations

import math

from core.contracts import ValuationAssumptions, ValuationResult


def gordon_growth_value(
    dividend_next_period: float,
    required_return: float,
    growth_rate: float,
) -> float:
    if not math.isfinite(dividend_next_period) or dividend_next_period < 0.0:
        raise ValueError("dividend_next_period must be finite and non-negative")
    _validate_discount_growth(required_return, growth_rate)
    return dividend_next_period / (required_return - growth_rate)


def gordon_result(
    dividend_next_period: float,
    assumptions: ValuationAssumptions,
) -> ValuationResult:
    required_return = _required_rate(assumptions)
    growth_rate = _growth_rate(assumptions)
    value = gordon_growth_value(dividend_next_period, required_return, growth_rate)
    return ValuationResult(
        method="gordon_growth",
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        basis="equity",
        unit="currency_per_share",
        value=value,
        equity_value=value,
        per_share_value=value,
    )


def ddm_value(
    dividends: tuple[float, ...],
    required_return: float,
    terminal_growth_rate: float,
) -> float:
    if not dividends:
        raise ValueError("at least one dividend forecast is required")
    if any(not math.isfinite(dividend) or dividend < 0.0 for dividend in dividends):
        raise ValueError("dividend forecasts must be finite and non-negative")
    _validate_discount_growth(required_return, terminal_growth_rate)
    present_value = sum(
        dividend / (1.0 + required_return) ** period
        for period, dividend in enumerate(dividends, start=1)
    )
    terminal_dividend = dividends[-1] * (1.0 + terminal_growth_rate)
    terminal_value = terminal_dividend / (required_return - terminal_growth_rate)
    return present_value + terminal_value / (1.0 + required_return) ** len(dividends)


def ddm_result(dividends: tuple[float, ...], assumptions: ValuationAssumptions) -> ValuationResult:
    required_return = _required_rate(assumptions)
    growth_rate = _growth_rate(assumptions)
    value = ddm_value(dividends, required_return, growth_rate)
    return ValuationResult(
        method="ddm",
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        basis="equity",
        unit="currency_per_share",
        value=value,
        equity_value=value,
        per_share_value=value,
    )


def _required_rate(assumptions: ValuationAssumptions) -> float:
    if assumptions.discount_rate is None:
        raise ValueError("discount_rate is required")
    return assumptions.discount_rate


def _growth_rate(assumptions: ValuationAssumptions) -> float:
    if assumptions.terminal_growth_rate is None:
        raise ValueError("terminal_growth_rate is required")
    return assumptions.terminal_growth_rate


def _validate_discount_growth(required_return: float, growth_rate: float) -> None:
    if not math.isfinite(required_return) or not 0.0 < required_return < 1.0:
        raise ValueError("required_return must be between 0 and 1")
    if not math.isfinite(growth_rate) or growth_rate >= required_return:
        raise ValueError("growth_rate must be finite and lower than required_return")
