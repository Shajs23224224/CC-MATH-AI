"""Deterministic risk-adjusted performance metrics for simple returns."""

from __future__ import annotations

import math
from collections.abc import Sequence

from quant.returns.performance import annualized_return, max_drawdown
from quant.returns.validation import validate_simple_returns


def beta(asset_returns: Sequence[float], benchmark_returns: Sequence[float]) -> float:
    """Estimate sample covariance(asset, benchmark) / sample variance(benchmark)."""
    asset, benchmark = _paired_returns(asset_returns, benchmark_returns)
    if len(asset) < 2:
        raise ValueError("at least two paired returns are required to estimate beta")
    asset_mean = math.fsum(asset) / len(asset)
    benchmark_mean = math.fsum(benchmark) / len(benchmark)
    benchmark_variance = math.fsum(
        (value - benchmark_mean) ** 2 for value in benchmark
    ) / (len(benchmark) - 1)
    if benchmark_variance == 0.0:
        raise ValueError("beta is undefined for a constant benchmark series")
    covariance = math.fsum(
        (asset_value - asset_mean) * (benchmark_value - benchmark_mean)
        for asset_value, benchmark_value in zip(asset, benchmark, strict=True)
    ) / (len(asset) - 1)
    return covariance / benchmark_variance


def treynor_ratio(
    returns: Sequence[float],
    benchmark_returns: Sequence[float],
    periods_per_year: float,
    risk_free_per_period: float = 0.0,
) -> float:
    """Return annualized arithmetic excess return divided by estimated beta."""
    checked, benchmark = _paired_returns(returns, benchmark_returns)
    _validate_periods_per_year(periods_per_year)
    _validate_finite(risk_free_per_period, "risk_free_per_period")
    estimated_beta = beta(checked, benchmark)
    if estimated_beta == 0.0:
        raise ValueError("Treynor ratio is undefined when beta is zero")
    excess_mean = math.fsum(value - risk_free_per_period for value in checked) / len(checked)
    return excess_mean * periods_per_year / estimated_beta


def calmar_ratio(returns: Sequence[float], periods_per_year: float) -> float:
    """Return geometric annualized return divided by maximum drawdown."""
    checked = validate_simple_returns(returns)
    _validate_periods_per_year(periods_per_year)
    drawdown = max_drawdown(checked)
    if drawdown == 0.0:
        raise ValueError("Calmar ratio is undefined when maximum drawdown is zero")
    return annualized_return(checked, periods_per_year) / drawdown


def omega_ratio(
    returns: Sequence[float],
    threshold_per_period: float = 0.0,
) -> float:
    """Return gains above a target divided by losses below that target."""
    checked = validate_simple_returns(returns)
    _validate_finite(threshold_per_period, "threshold_per_period")
    gains = math.fsum(max(value - threshold_per_period, 0.0) for value in checked)
    losses = math.fsum(max(threshold_per_period - value, 0.0) for value in checked)
    if losses == 0.0:
        raise ValueError("Omega ratio is undefined when there are no losses below threshold")
    return gains / losses


def information_ratio(
    returns: Sequence[float],
    benchmark_returns: Sequence[float],
    periods_per_year: float,
) -> float:
    """Return annualized mean active return divided by active-return sample volatility."""
    asset, benchmark = _paired_returns(returns, benchmark_returns)
    _validate_periods_per_year(periods_per_year)
    active = tuple(
        asset_value - benchmark_value
        for asset_value, benchmark_value in zip(asset, benchmark, strict=True)
    )
    if len(active) < 2:
        raise ValueError("at least two paired returns are required for information ratio")
    mean_active = math.fsum(active) / len(active)
    variance_active = math.fsum((value - mean_active) ** 2 for value in active) / (
        len(active) - 1
    )
    if variance_active == 0.0:
        raise ValueError("information ratio is undefined when active-return volatility is zero")
    return mean_active / math.sqrt(variance_active) * math.sqrt(periods_per_year)


def _paired_returns(
    asset_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    asset = validate_simple_returns(asset_returns)
    benchmark = validate_simple_returns(benchmark_returns)
    if len(asset) != len(benchmark):
        raise ValueError("asset and benchmark return series must have equal length")
    return asset, benchmark


def _validate_periods_per_year(periods_per_year: float) -> None:
    _validate_finite(periods_per_year, "periods_per_year")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")


def _validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
