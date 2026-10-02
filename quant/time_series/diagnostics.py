"""Deterministic residual diagnostics for time-series models."""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.stats import chi2

from core.contracts import ResidualDiagnostics

from ..statistics import autocorrelation
from .validation import validate_univariate


def ljung_box_test(residuals: Sequence[float], lag: int = 10) -> tuple[float, float]:
    """Return Ljung-Box Q statistic and p-value."""
    checked = validate_univariate(residuals, minimum=3)
    max_lag = min(lag, len(checked) - 2)
    if max_lag < 1:
        raise ValueError("residual sample is too short for Ljung-Box diagnostics")
    mean = math.fsum(checked) / len(checked)
    centered = tuple(value - mean for value in checked)
    denominator = math.fsum(value * value for value in centered)
    if denominator == 0.0:
        raise ValueError("Ljung-Box diagnostic is undefined for zero-variance residuals")
    statistic = 0.0
    for current_lag in range(1, max_lag + 1):
        rho = (
            math.fsum(
                centered[index] * centered[index - current_lag]
                for index in range(current_lag, len(centered))
            )
            / denominator
        )
        statistic += rho * rho / (len(centered) - current_lag)
    statistic *= len(centered) * (len(centered) + 2.0)
    pvalue = float(chi2.sf(statistic, max_lag))
    return statistic, pvalue


def residual_diagnostics(
    residuals: Sequence[float],
    lag: int = 10,
) -> ResidualDiagnostics:
    """Summarize residual variance, autocorrelation and Ljung-Box statistics."""
    checked = validate_univariate(residuals, minimum=3)
    statistic, pvalue = ljung_box_test(checked, lag=lag)
    mean = math.fsum(checked) / len(checked)
    variance = math.fsum((value - mean) ** 2 for value in checked) / len(checked)
    rho = autocorrelation(checked, 1) if variance > 0.0 else 0.0
    return ResidualDiagnostics(
        observations=len(checked),
        mean=mean,
        variance=variance,
        lag=min(lag, len(checked) - 2),
        ljung_box_statistic=statistic,
        ljung_box_pvalue=pvalue,
        residual_autocorrelation=rho,
    )
