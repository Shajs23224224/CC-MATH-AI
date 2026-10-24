"""Reproducible bootstrap distributions for F41."""

from __future__ import annotations

import random
from collections.abc import Sequence
from math import ceil, isfinite
from statistics import mean, median

from backtest.engine import BacktestResult
from backtest.robustness.contracts import MonteCarloConfig, MonteCarloSummary


def _period_returns(result: BacktestResult) -> tuple[float, ...]:
    equity = tuple(point.equity for point in result.equity_curve)
    if len(equity) < 2:
        raise ValueError("Monte Carlo requires at least two equity observations")
    if any(not isfinite(value) or value <= 0 for value in equity):
        raise ValueError("Monte Carlo bootstrap requires a strictly positive equity curve")
    returns = tuple(
        current / previous - 1.0 for previous, current in zip(equity, equity[1:], strict=False)
    )
    if not returns or not all(isfinite(value) for value in returns):
        raise ValueError("observed equity returns must be finite and non-empty")
    return returns


def _draw_sample(
    returns: tuple[float, ...],
    config: MonteCarloConfig,
    rng: random.Random,
) -> tuple[float, ...]:
    length = len(returns)
    if config.method == "iid":
        return tuple(returns[rng.randrange(length)] for _ in range(length))

    sample: list[float] = []
    while len(sample) < length:
        start = rng.randrange(length)
        stop = min(start + config.block_size, length)
        sample.extend(returns[start:stop])
    return tuple(sample[:length])


def _simulate_path(returns: tuple[float, ...]) -> tuple[float, float]:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for period_return in returns:
        growth = 1.0 + period_return
        # Treat a total loss or worse as bankruptcy; equity cannot recover afterward.
        equity = 0.0 if growth <= 0 else equity * growth
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return equity - 1.0, max_drawdown


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run_monte_carlo(
    result: BacktestResult,
    config: MonteCarloConfig | None = None,
) -> MonteCarloSummary:
    """Bootstrap observed mark-to-market period returns with a fixed local RNG seed.

    This resamples the realized return sequence; it does not retrain the strategy or
    simulate market microstructure. The moving-block method preserves short local runs
    of observed returns, but neither method removes selection bias from the original run.
    """
    settings = config or MonteCarloConfig()
    observed = _period_returns(result)
    rng = random.Random(settings.seed)
    terminal_returns: list[float] = []
    drawdowns: list[float] = []
    for _ in range(settings.simulations):
        sample = _draw_sample(observed, settings, rng)
        terminal_return, max_drawdown = _simulate_path(sample)
        terminal_returns.append(terminal_return)
        drawdowns.append(max_drawdown)

    ordered = sorted(terminal_returns)
    tail_count = max(1, ceil((1.0 - settings.confidence_level) * len(ordered)))
    return MonteCarloSummary(
        simulations=settings.simulations,
        seed=settings.seed,
        method=settings.method,
        block_size=settings.block_size,
        confidence_level=settings.confidence_level,
        mean_return=mean(terminal_returns),
        median_return=median(terminal_returns),
        lower_quantile_return=_percentile(terminal_returns, 1.0 - settings.confidence_level),
        upper_quantile_return=_percentile(terminal_returns, settings.confidence_level),
        probability_of_loss=sum(value < 0 for value in terminal_returns) / len(terminal_returns),
        expected_shortfall=mean(ordered[:tail_count]),
        p95_max_drawdown=_percentile(tuple(drawdowns), 0.95),
    )
