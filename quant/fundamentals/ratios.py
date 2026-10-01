"""Deterministic fundamental-analysis ratios and margins."""

from __future__ import annotations

import math

from core.contracts import FundamentalSnapshot


def _require(snapshot: FundamentalSnapshot, field: str) -> float:
    value = getattr(snapshot, field)
    if value is None or not math.isfinite(value):
        raise ValueError(f"{field} is required and must be finite")
    return float(value)


def _ratio(numerator: float, denominator: float, name: str) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError(f"{name} inputs must be finite")
    if denominator == 0.0:
        raise ValueError(f"{name} denominator cannot be zero")
    return numerator / denominator


def _positive_denominator(value: float, name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    return value


def roe(snapshot: FundamentalSnapshot) -> float:
    """Return return on equity: net income / equity."""
    return _ratio(_require(snapshot, "net_income"), _require(snapshot, "equity"), "ROE")


def roa(snapshot: FundamentalSnapshot) -> float:
    """Return return on assets: net income / total assets."""
    assets = _positive_denominator(_require(snapshot, "total_assets"), "ROA")
    return _ratio(_require(snapshot, "net_income"), assets, "ROA")


def roic(snapshot: FundamentalSnapshot, tax_rate: float) -> float:
    """Return ROIC using NOPAT / invested capital.

    NOPAT = EBIT * (1 - tax_rate)
    Invested capital = equity + debt - cash
    """
    if not math.isfinite(tax_rate):
        raise ValueError("tax_rate must be finite")
    invested_capital = (
        _require(snapshot, "equity") + _require(snapshot, "debt") - _require(snapshot, "cash")
    )
    invested_capital = _positive_denominator(invested_capital, "ROIC")
    nopat = _require(snapshot, "ebit") * (1.0 - tax_rate)
    return _ratio(nopat, invested_capital, "ROIC")


def roce(snapshot: FundamentalSnapshot) -> float:
    """Return ROCE: EBIT / (total assets - current liabilities)."""
    capital_employed = _require(snapshot, "total_assets") - _require(
        snapshot, "current_liabilities"
    )
    capital_employed = _positive_denominator(capital_employed, "ROCE")
    return _ratio(_require(snapshot, "ebit"), capital_employed, "ROCE")


def ebitda_margin(snapshot: FundamentalSnapshot) -> float:
    """Return EBITDA margin: EBITDA / revenue."""
    return _ratio(_require(snapshot, "ebitda"), _require(snapshot, "revenue"), "EBITDA margin")


def ebit_margin(snapshot: FundamentalSnapshot) -> float:
    """Return EBIT margin: EBIT / revenue."""
    return _ratio(_require(snapshot, "ebit"), _require(snapshot, "revenue"), "EBIT margin")


def net_margin(snapshot: FundamentalSnapshot) -> float:
    """Return net margin: net income / revenue."""
    return _ratio(_require(snapshot, "net_income"), _require(snapshot, "revenue"), "net margin")


def free_cash_flow_margin(snapshot: FundamentalSnapshot) -> float:
    """Return FCF margin: free cash flow / revenue."""
    return _ratio(
        _require(snapshot, "free_cash_flow"),
        _require(snapshot, "revenue"),
        "FCF margin",
    )


def cash_conversion_ratio(snapshot: FundamentalSnapshot) -> float:
    """Return cash conversion as free cash flow / net income."""
    return _ratio(
        _require(snapshot, "free_cash_flow"),
        _require(snapshot, "net_income"),
        "cash conversion",
    )


def payout_ratio(snapshot: FundamentalSnapshot) -> float:
    """Return dividend payout ratio: dividends / net income."""
    return _ratio(_require(snapshot, "dividends"), _require(snapshot, "net_income"), "payout ratio")


def retention_ratio(snapshot: FundamentalSnapshot) -> float:
    """Return earnings retention: 1 - payout ratio."""
    return 1.0 - payout_ratio(snapshot)


def debt_to_equity(snapshot: FundamentalSnapshot) -> float:
    """Return leverage as debt / equity."""
    return _ratio(_require(snapshot, "debt"), _require(snapshot, "equity"), "debt-to-equity")


def debt_to_assets(snapshot: FundamentalSnapshot) -> float:
    """Return debt / total assets."""
    assets = _positive_denominator(_require(snapshot, "total_assets"), "debt-to-assets")
    return _ratio(_require(snapshot, "debt"), assets, "debt-to-assets")


def interest_coverage(snapshot: FundamentalSnapshot) -> float:
    """Return EBIT / interest expense."""
    return _ratio(
        _require(snapshot, "ebit"),
        _require(snapshot, "interest_expense"),
        "interest coverage",
    )
