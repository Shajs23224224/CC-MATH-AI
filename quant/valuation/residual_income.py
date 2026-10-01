"""Residual-income valuation."""

from __future__ import annotations

import math

from core.contracts import ValuationAssumptions, ValuationResult


def residual_income_value(
    book_value_per_share: float,
    residual_incomes: tuple[float, ...],
    required_return: float,
    terminal_growth_rate: float,
) -> float:
    book_value = _positive(book_value_per_share, "book_value_per_share")
    del book_value
    if not residual_incomes:
        raise ValueError("at least one residual-income forecast is required")
    if any(not math.isfinite(value) for value in residual_incomes):
        raise ValueError("residual incomes must be finite")
    _validate_rates(required_return, terminal_growth_rate)

    present_value = sum(
        value / (1.0 + required_return) ** period
        for period, value in enumerate(residual_incomes, start=1)
    )
    terminal_income = residual_incomes[-1] * (1.0 + terminal_growth_rate)
    terminal_value = terminal_income / (required_return - terminal_growth_rate)
    return (
        book_value_per_share
        + present_value
        + terminal_value / (1.0 + required_return) ** len(residual_incomes)
    )


def residual_income_result(
    book_value_per_share: float,
    residual_incomes: tuple[float, ...],
    assumptions: ValuationAssumptions,
) -> ValuationResult:
    required_return = _required(assumptions.discount_rate, "discount_rate")
    growth = _required(assumptions.terminal_growth_rate, "terminal_growth_rate")
    value = residual_income_value(
        book_value_per_share,
        residual_incomes,
        required_return,
        growth,
    )
    return ValuationResult(
        method="residual_income",
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        basis="equity",
        unit="currency_per_share",
        value=value,
        equity_value=value,
        per_share_value=value,
    )


def _required(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(value):
        raise ValueError(f"{name} is required and must be finite")
    return value


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_rates(required_return: float, growth: float) -> None:
    if not 0.0 < required_return < 1.0:
        raise ValueError("required_return must be between 0 and 1")
    if not -1.0 < growth < required_return:
        raise ValueError("terminal_growth_rate must be lower than required_return")
