"""Deterministic statistical engine."""

from .dependence import autocorrelation, rankdata, spearman_correlation
from .descriptive import (
    descriptive_summary,
    excess_kurtosis,
    interquartile_range,
    median_absolute_deviation,
    quantile,
    skewness,
)
from .rolling import rolling_mean, rolling_std, rolling_zscore
from .validation import validate_sample

__all__ = [
    "autocorrelation",
    "descriptive_summary",
    "excess_kurtosis",
    "interquartile_range",
    "median_absolute_deviation",
    "quantile",
    "rankdata",
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "skewness",
    "spearman_correlation",
    "validate_sample",
]
