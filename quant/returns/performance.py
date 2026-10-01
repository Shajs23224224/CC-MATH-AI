"""Deterministic investment performance metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.contracts import PerformanceMetrics
from quant.math import sample_std

from .calculations import cumulative_return
from .validation import validate_simple_returns


def wealth_index(returns: Sequence[float], initial_value: float = 1.0) -> tuple[float, ...]:
    """Build a wealth index beginning at initial_value."""
    checked = validate_simple_returns(returns)
    if not math.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial_value must be finite and positive.")

    values = [initial_value]
    wealth = initial_value
    for value in checked:
        wealth *= 1.0 + value
        values.append(wealth)
    return tuple(values)


def drawdown_series(returns: Sequence[float]) -> tuple[float, ...]:
    """Return percentage drawdown from the running wealth maximum."""
    wealth_values = wealth_index(returns)
    running_max = wealth_values[0]
    drawdowns = []
    for wealth in wealth_values:
        running_max = max(running_max, wealth)
        drawdowns.append(wealth / running_max - 1.0)
    return tuple(drawdowns)


def max_drawdown(returns: Sequence[float]) -> float:
    """Return maximum peak-to-trough drawdown as a positive fraction."""
    return max(0.0, -min(drawdown_series(returns)))


def annualized_return(returns: Sequence[float], periods_per_year: float) -> float:
    """Return the geometric annualized return."""
    checked = validate_simple_returns(returns)
    _validate_periods_per_year(periods_per_year)
    log_growth = math.fsum(math.log1p(value) for value in checked)
    return math.expm1(log_growth * periods_per_year / len(checked))


def annualized_volatility(returns: Sequence[float], periods_per_year: float) -> float:
    """Return sample volatility scaled to periods_per_year."""
    checked = validate_simple_returns(returns)
    _validate_periods_per_year(periods_per_year)
    if len(checked) < 2:
        raise ValueError("at least two returns are required for volatility.")
    return sample_std(checked) * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: Sequence[float],
    periods_per_year: float,
    risk_free_per_period: float = 0.0,
) -> float:
    """Return the annualized Sharpe ratio using periodic risk-free return."""
    checked = validate_simple_returns(returns)
    _validate_periods_per_year(periods_per_year)
    if not math.isfinite(risk_free_per_period):
        raise ValueError("risk_free_per_period must be finite.")
    excess = tuple(value - risk_free_per_period for value in checked)
    if len(excess) < 2:
        raise ValueError("at least two returns are required for Sharpe ratio.")
    volatility = sample_std(excess)
    if volatility == 0.0:
        raise ValueError("Sharpe ratio is undefined for zero excess-return volatility.")
    return math.fsum(excess) / len(excess) / volatility * math.sqrt(periods_per_year)


def downside_deviation(
    returns: Sequence[float],
    target_return_per_period: float = 0.0,
) -> float:
    """Return root-mean-square downside deviation below a periodic target."""
    checked = validate_simple_returns(returns)
    if not math.isfinite(target_return_per_period):
        raise ValueError("target_return_per_period must be finite.")
    squared_downside = math.fsum(
        min(0.0, value - target_return_per_period) ** 2 for value in checked
    )
    return math.sqrt(squared_downside / len(checked))


def sortino_ratio(
    returns: Sequence[float],
    periods_per_year: float,
    target_return_per_period: float = 0.0,
) -> float:
    """Return an annualized Sortino ratio."""
    checked = validate_simple_returns(returns)
    _validate_periods_per_year(periods_per_year)
    downside = downside_deviation(checked, target_return_per_period)
    if downside == 0.0:
        raise ValueError("Sortino ratio is undefined with zero downside deviation.")
    excess_mean = math.fsum(value - target_return_per_period for value in checked) / len(checked)
    return excess_mean / downside * math.sqrt(periods_per_year)


def positive_period_ratio(returns: Sequence[float]) -> float:
    """Return the fraction of observations with positive return."""
    checked = validate_simple_returns(returns)
    return sum(value > 0.0 for value in checked) / len(checked)


def performance_metrics(
    returns: Sequence[float],
    periods_per_year: float,
    *,
    risk_free_per_period: float = 0.0,
    target_return_per_period: float = 0.0,
) -> PerformanceMetrics:
    """Compute the complete F14 performance metric set."""
    checked = validate_simple_returns(returns)
    return PerformanceMetrics(
        observations=len(checked),
        periods_per_year=periods_per_year,
        total_return=cumulative_return(checked),
        annualized_return=annualized_return(checked, periods_per_year),
        annualized_volatility=annualized_volatility(checked, periods_per_year),
        sharpe_ratio=sharpe_ratio(
            checked,
            periods_per_year,
            risk_free_per_period,
        ),
        sortino_ratio=sortino_ratio(
            checked,
            periods_per_year,
            target_return_per_period,
        ),
        max_drawdown=max_drawdown(checked),
        positive_period_ratio=positive_period_ratio(checked),
    )


def _validate_periods_per_year(periods_per_year: float) -> None:
    if not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be finite and positive.")
