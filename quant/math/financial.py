"""Foundational deterministic time-value-of-money functions."""

from __future__ import annotations

import math


def _require_finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all numeric inputs must be finite.")


def discount_factor(rate_per_period: float, periods: int) -> float:
    """Return 1/(1+r)^n for discrete discounting."""
    _require_finite(rate_per_period)
    if periods < 0:
        raise ValueError("periods must be non-negative.")
    if 1.0 + rate_per_period <= 0.0:
        raise ValueError("rate_per_period must be greater than -100%.")
    return math.pow(1.0 + rate_per_period, -periods)


def present_value(future_value_amount: float, rate_per_period: float, periods: int) -> float:
    """Discount a single future cash flow to present value."""
    _require_finite(future_value_amount, rate_per_period)
    return future_value_amount * discount_factor(rate_per_period, periods)


def future_value(present_value_amount: float, rate_per_period: float, periods: int) -> float:
    """Compound a single present cash flow into the future."""
    _require_finite(present_value_amount, rate_per_period)
    if periods < 0:
        raise ValueError("periods must be non-negative.")
    if 1.0 + rate_per_period <= 0.0:
        raise ValueError("rate_per_period must be greater than -100%.")
    return present_value_amount * math.pow(1.0 + rate_per_period, periods)


def annuity_present_value(
    payment: float,
    rate_per_period: float,
    periods: int,
) -> float:
    """Return the present value of an ordinary annuity."""
    _require_finite(payment, rate_per_period)
    if periods <= 0:
        raise ValueError("periods must be positive.")
    if 1.0 + rate_per_period <= 0.0:
        raise ValueError("rate_per_period must be greater than -100%.")
    if rate_per_period == 0.0:
        return payment * periods
    return payment * (1.0 - discount_factor(rate_per_period, periods)) / rate_per_period


def annuity_future_value(
    payment: float,
    rate_per_period: float,
    periods: int,
) -> float:
    """Return the future value of an ordinary annuity."""
    _require_finite(payment, rate_per_period)
    if periods <= 0:
        raise ValueError("periods must be positive.")
    if 1.0 + rate_per_period <= 0.0:
        raise ValueError("rate_per_period must be greater than -100%.")
    if rate_per_period == 0.0:
        return payment * periods
    return payment * (math.pow(1.0 + rate_per_period, periods) - 1.0) / rate_per_period


def annuity_payment(
    present_value_amount: float,
    rate_per_period: float,
    periods: int,
) -> float:
    """Return the ordinary annuity payment required for a target present value."""
    _require_finite(present_value_amount, rate_per_period)
    if periods <= 0:
        raise ValueError("periods must be positive.")
    if 1.0 + rate_per_period <= 0.0:
        raise ValueError("rate_per_period must be greater than -100%.")
    if rate_per_period == 0.0:
        return present_value_amount / periods
    factor = rate_per_period / (1.0 - discount_factor(rate_per_period, periods))
    return present_value_amount * factor
