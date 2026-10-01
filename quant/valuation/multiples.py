"""Deterministic relative valuation metrics."""

from __future__ import annotations

import math

from core.contracts import ValuationAssumptions, ValuationResult


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def pe_ratio(price: float, earnings_per_share: float) -> float:
    return _positive(price, "price") / _positive(earnings_per_share, "earnings_per_share")


def ps_ratio(price: float, revenue_per_share: float) -> float:
    return _positive(price, "price") / _positive(revenue_per_share, "revenue_per_share")


def pb_ratio(price: float, book_value_per_share: float) -> float:
    return _positive(price, "price") / _positive(book_value_per_share, "book_value_per_share")


def ev_to_ebitda(
    market_capitalization: float,
    debt: float,
    cash: float,
    ebitda: float,
) -> float:
    return _enterprise_value(market_capitalization, debt, cash) / _positive(ebitda, "ebitda")


def ev_to_fcf(
    market_capitalization: float,
    debt: float,
    cash: float,
    free_cash_flow: float,
) -> float:
    return _enterprise_value(market_capitalization, debt, cash) / _positive(
        free_cash_flow, "free_cash_flow"
    )


def earnings_yield(price: float, earnings_per_share: float) -> float:
    return _positive(earnings_per_share, "earnings_per_share") / _positive(price, "price")


def implied_equity_value_from_multiple(
    multiple: float,
    metric_value: float,
    *,
    debt: float = 0.0,
    cash: float = 0.0,
    basis: str = "equity",
) -> float:
    multiple = _finite(multiple, "multiple")
    metric_value = _positive(metric_value, "metric_value")
    debt = _non_negative(debt, "debt")
    cash = _non_negative(cash, "cash")
    if basis == "equity":
        return multiple * metric_value
    if basis == "enterprise":
        enterprise_value = multiple * metric_value
        return enterprise_value - debt + cash
    raise ValueError("basis must be 'equity' or 'enterprise'")


def comparable_per_share_value(
    multiple: float,
    metric_per_share: float,
) -> float:
    return _finite(multiple, "multiple") * _positive(metric_per_share, "metric_per_share")


def _enterprise_value(market_capitalization: float, debt: float, cash: float) -> float:
    return (
        _positive(market_capitalization, "market_capitalization")
        + _non_negative(debt, "debt")
        - _non_negative(cash, "cash")
    )


def _non_negative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return value


def multiple_result(
    method: str,
    multiple: float,
    assumptions: ValuationAssumptions,
) -> ValuationResult:
    return ValuationResult(
        method=method,
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        basis="equity",
        unit="multiple",
        value=_finite(multiple, "multiple"),
    )
