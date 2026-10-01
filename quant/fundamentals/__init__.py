"""Deterministic fundamental-analysis algorithms."""

from .growth import fundamental_growth, growth_rate
from .ratios import (
    cash_conversion_ratio,
    debt_to_assets,
    debt_to_equity,
    ebit_margin,
    ebitda_margin,
    free_cash_flow_margin,
    interest_coverage,
    net_margin,
    payout_ratio,
    retention_ratio,
    roa,
    roce,
    roe,
    roic,
)

__all__ = [
    "cash_conversion_ratio",
    "debt_to_assets",
    "debt_to_equity",
    "ebit_margin",
    "ebitda_margin",
    "free_cash_flow_margin",
    "fundamental_growth",
    "growth_rate",
    "interest_coverage",
    "net_margin",
    "payout_ratio",
    "retention_ratio",
    "roa",
    "roce",
    "roe",
    "roic",
]
