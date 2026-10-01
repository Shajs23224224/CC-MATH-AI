"""Foundational deterministic algebra operations."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _finite_values(values: Sequence[float], *, name: str = "values") -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def arithmetic_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean using compensated summation."""
    checked = _finite_values(values)
    return math.fsum(checked) / len(checked)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return a weighted arithmetic mean."""
    checked_values = _finite_values(values, name="values")
    checked_weights = _finite_values(weights, name="weights")
    if len(checked_values) != len(checked_weights):
        raise ValueError("values and weights must have the same length.")
    weight_sum = math.fsum(checked_weights)
    if weight_sum == 0.0:
        raise ValueError("weights must not sum to zero.")
    weighted_total = math.fsum(
        v * w for v, w in zip(checked_values, checked_weights, strict=True)
    )
    return weighted_total / weight_sum


def geometric_mean(values: Sequence[float]) -> float:
    """Return the geometric mean for strictly positive values."""
    checked = _finite_values(values)
    if any(value <= 0.0 for value in checked):
        raise ValueError("geometric_mean requires strictly positive values.")
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def harmonic_mean(values: Sequence[float]) -> float:
    """Return the harmonic mean for strictly positive values."""
    checked = _finite_values(values)
    if any(value <= 0.0 for value in checked):
        raise ValueError("harmonic_mean requires strictly positive values.")
    reciprocal_sum = math.fsum(1.0 / value for value in checked)
    return len(checked) / reciprocal_sum


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the scalar product of two equal-length finite sequences."""
    left_checked = _finite_values(left, name="left")
    right_checked = _finite_values(right, name="right")
    if len(left_checked) != len(right_checked):
        raise ValueError("left and right must have the same length.")
    return math.fsum(
        x * y for x, y in zip(left_checked, right_checked, strict=True)
    )


def compound_growth(initial_value: float, rate_per_period: float, periods: int) -> float:
    """Apply deterministic discrete compounding over a non-negative period count."""
    if not math.isfinite(initial_value) or not math.isfinite(rate_per_period):
        raise ValueError("initial_value and rate_per_period must be finite.")
    if periods < 0:
        raise ValueError("periods must be non-negative.")
    if 1.0 + rate_per_period <= 0.0:
        raise ValueError("rate_per_period must be greater than -100%.")
    return initial_value * math.pow(1.0 + rate_per_period, periods)
