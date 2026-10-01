"""Deterministic return calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.contracts import PriceSeries, ReturnSeries

from .validation import validate_simple_returns


def simple_return(previous_price: float, current_price: float) -> float:
    """Return the simple holding-period return."""
    if not math.isfinite(previous_price) or not math.isfinite(current_price):
        raise ValueError("prices must be finite.")
    if previous_price <= 0.0 or current_price <= 0.0:
        raise ValueError("prices must be positive.")
    return current_price / previous_price - 1.0


def log_return(previous_price: float, current_price: float) -> float:
    """Return the continuously compounded holding-period return."""
    if not math.isfinite(previous_price) or not math.isfinite(current_price):
        raise ValueError("prices must be finite.")
    if previous_price <= 0.0 or current_price <= 0.0:
        raise ValueError("prices must be positive.")
    return math.log(current_price / previous_price)


def simple_returns(prices: Sequence[float]) -> tuple[float, ...]:
    """Calculate simple returns from a positive price sequence."""
    if len(prices) < 2:
        raise ValueError("at least two prices are required.")
    checked = tuple(float(price) for price in prices)
    if not all(math.isfinite(price) and price > 0.0 for price in checked):
        raise ValueError("prices must be finite and positive.")
    return tuple(current / previous - 1.0 for previous, current in zip(checked, checked[1:]))


def log_returns(prices: Sequence[float]) -> tuple[float, ...]:
    """Calculate log returns from a positive price sequence."""
    if len(prices) < 2:
        raise ValueError("at least two prices are required.")
    checked = tuple(float(price) for price in prices)
    if not all(math.isfinite(price) and price > 0.0 for price in checked):
        raise ValueError("prices must be finite and positive.")
    return tuple(math.log(current / previous) for previous, current in zip(checked, checked[1:]))


def simple_return_series(price_series: PriceSeries) -> ReturnSeries:
    """Transform a PriceSeries into timestamp-aligned simple returns."""
    return ReturnSeries(
        asset=price_series.asset,
        timestamps=price_series.timestamps[1:],
        values=simple_returns(price_series.values),
        frequency=price_series.frequency,
        return_type="simple",
    )


def log_return_series(price_series: PriceSeries) -> ReturnSeries:
    """Transform a PriceSeries into timestamp-aligned log returns."""
    return ReturnSeries(
        asset=price_series.asset,
        timestamps=price_series.timestamps[1:],
        values=log_returns(price_series.values),
        frequency=price_series.frequency,
        return_type="log",
    )


def cumulative_return(returns: Sequence[float]) -> float:
    """Compound simple returns into a cumulative return."""
    checked = validate_simple_returns(returns)
    log_growth = math.fsum(math.log1p(value) for value in checked)
    return math.expm1(log_growth)
