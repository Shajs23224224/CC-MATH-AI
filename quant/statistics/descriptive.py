"""Deterministic descriptive and robust statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.contracts import StatisticalSummary

from .validation import validate_sample


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated sample quantile in [0, 1]."""
    checked = sorted(validate_sample(values))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between 0 and 1")
    if len(checked) == 1:
        return checked[0]
    position = probability * (len(checked) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return checked[lower]
    weight = position - lower
    return checked[lower] + weight * (checked[upper] - checked[lower])


def interquartile_range(values: Sequence[float]) -> float:
    """Return Q3 - Q1."""
    return quantile(values, 0.75) - quantile(values, 0.25)


def median_absolute_deviation(values: Sequence[float]) -> float:
    """Return the median absolute deviation around the sample median."""
    checked = validate_sample(values)
    center = quantile(checked, 0.5)
    return quantile(tuple(abs(value - center) for value in checked), 0.5)


def skewness(values: Sequence[float]) -> float:
    """Return the standardized third central moment."""
    checked = validate_sample(values, minimum=2)
    mean = math.fsum(checked) / len(checked)
    second = math.fsum((value - mean) ** 2 for value in checked) / len(checked)
    if second == 0.0:
        raise ValueError("skewness is undefined for a zero-variance sample")
    third = math.fsum((value - mean) ** 3 for value in checked) / len(checked)
    return third / second**1.5


def excess_kurtosis(values: Sequence[float]) -> float:
    """Return Pearson kurtosis minus three (the normal distribution is zero)."""
    checked = validate_sample(values, minimum=2)
    mean = math.fsum(checked) / len(checked)
    second = math.fsum((value - mean) ** 2 for value in checked) / len(checked)
    if second == 0.0:
        raise ValueError("kurtosis is undefined for a zero-variance sample")
    fourth = math.fsum((value - mean) ** 4 for value in checked) / len(checked)
    return fourth / second**2 - 3.0


def descriptive_summary(values: Sequence[float]) -> StatisticalSummary:
    """Return a deterministic descriptive-statistics summary."""
    checked = validate_sample(values)
    ordered = sorted(checked)
    mean = math.fsum(checked) / len(checked)
    variance = math.fsum((value - mean) ** 2 for value in checked) / len(checked)
    q1 = quantile(checked, 0.25)
    q3 = quantile(checked, 0.75)
    return StatisticalSummary(
        observations=len(checked),
        mean=mean,
        median=quantile(checked, 0.5),
        variance=variance,
        standard_deviation=math.sqrt(variance),
        skewness=skewness(checked) if len(checked) >= 2 and variance > 0.0 else 0.0,
        excess_kurtosis=excess_kurtosis(checked) if len(checked) >= 2 and variance > 0.0 else 0.0,
        minimum=ordered[0],
        maximum=ordered[-1],
        first_quartile=q1,
        third_quartile=q3,
        interquartile_range=q3 - q1,
        median_absolute_deviation=median_absolute_deviation(checked),
    )
