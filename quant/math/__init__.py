"""Foundational deterministic mathematical primitives."""

from .algebra import (
    arithmetic_mean,
    compound_growth,
    dot_product,
    geometric_mean,
    harmonic_mean,
    weighted_mean,
)
from .financial import (
    annuity_future_value,
    annuity_payment,
    annuity_present_value,
    discount_factor,
    future_value,
    present_value,
)
from .interpolation import linear_interpolate, linear_interpolate_series
from .optimization import golden_section_minimize
from .probability import binomial_pmf, normal_cdf, normal_pdf
from .statistics import (
    correlation,
    population_std,
    population_variance,
    sample_covariance,
    sample_std,
    sample_variance,
)

__all__ = [
    "annuity_future_value",
    "annuity_payment",
    "annuity_present_value",
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
    "population_std",
    "population_variance",
    "present_value",
    "sample_covariance",
    "sample_std",
    "sample_variance",
    "weighted_mean",
]
