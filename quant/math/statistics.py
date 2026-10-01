"""Foundational deterministic statistical operations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .algebra import _finite_values


def population_variance(values: Sequence[float]) -> float:
    """Return the population variance."""
    checked = _finite_values(values)
    mean = math.fsum(checked) / len(checked)
    return math.fsum((value - mean) ** 2 for value in checked) / len(checked)


def sample_variance(values: Sequence[float]) -> float:
    """Return the unbiased sample variance with Bessel correction."""
    checked = _finite_values(values)
    if len(checked) < 2:
        raise ValueError("sample_variance requires at least two observations.")
    mean = math.fsum(checked) / len(checked)
    return math.fsum((value - mean) ** 2 for value in checked) / (len(checked) - 1)


def population_std(values: Sequence[float]) -> float:
    """Return the population standard deviation."""
    return math.sqrt(population_variance(values))


def sample_std(values: Sequence[float]) -> float:
    """Return the sample standard deviation."""
    return math.sqrt(sample_variance(values))


def sample_covariance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return sample covariance between two equal-length sequences."""
    left_checked = _finite_values(left, name="left")
    right_checked = _finite_values(right, name="right")
    if len(left_checked) != len(right_checked):
        raise ValueError("left and right must have the same length.")
    if len(left_checked) < 2:
        raise ValueError("sample_covariance requires at least two observations.")

    left_mean = math.fsum(left_checked) / len(left_checked)
    right_mean = math.fsum(right_checked) / len(right_checked)
    return math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left_checked, right_checked, strict=True)
    ) / (len(left_checked) - 1)


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Pearson sample correlation."""
    covariance = sample_covariance(left, right)
    left_std = sample_std(left)
    right_std = sample_std(right)
    if left_std == 0.0 or right_std == 0.0:
        raise ValueError("correlation is undefined for a zero-variance series.")
    return covariance / (left_std * right_std)
