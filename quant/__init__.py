"""Deterministic quantitative and financial algorithms.

The package intentionally contains no model-generated arithmetic. Quantitative
functions are deterministic and reusable by later signal, portfolio, risk,
and backtesting layers.
"""

from .math import (
    arithmetic_mean,
    binomial_pmf,
    compound_growth,
    correlation,
    discount_factor,
    dot_product,
    future_value,
    geometric_mean,
    golden_section_minimize,
    harmonic_mean,
    linear_interpolate,
    linear_interpolate_series,
    normal_cdf,
    normal_pdf,
    present_value,
    sample_covariance,
    sample_std,
    sample_variance,
    weighted_mean,
)

__all__ = [
    "arithmetic_mean",
    "binomial_pmf",
    "compound_growth",
    "correlation",
    "discount_factor",
    "dot_product",
    "future_value",
    "geometric_mean",
    "golden_section_minimize",
    "harmonic_mean",
    "linear_interpolate",
    "linear_interpolate_series",
    "normal_cdf",
    "normal_pdf",
    "present_value",
    "sample_covariance",
    "sample_std",
    "sample_variance",
    "weighted_mean",
]
