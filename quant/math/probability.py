"""Foundational deterministic probability functions."""

from __future__ import annotations

import math


def normal_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Return the probability density of a normal distribution."""
    if not all(math.isfinite(value) for value in (x, mean, std)):
        raise ValueError("x, mean and std must be finite.")
    if std <= 0.0:
        raise ValueError("std must be positive.")
    standardized = (x - mean) / std
    numerator = math.exp(-0.5 * standardized * standardized)
    return numerator / (std * math.sqrt(2.0 * math.pi))


def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Return the cumulative distribution function of a normal distribution."""
    if not all(math.isfinite(value) for value in (x, mean, std)):
        raise ValueError("x, mean and std must be finite.")
    if std <= 0.0:
        raise ValueError("std must be positive.")
    standardized = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * math.erfc(-standardized)


def binomial_pmf(n: int, k: int, probability: float) -> float:
    """Return P(X=k) for X~Binomial(n, probability), stably for large n."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n.")
    if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
        raise ValueError("probability must be finite and lie in [0, 1].")
    if probability in (0.0, 1.0):
        return 1.0 if (probability == 1.0 and k == n) or (probability == 0.0 and k == 0) else 0.0

    log_combination = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    log_probability = log_combination
    log_probability += k * math.log(probability)
    log_probability += (n - k) * math.log1p(-probability)
    return min(1.0, max(0.0, math.exp(log_probability)))
