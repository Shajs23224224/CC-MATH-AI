"""Empirical probabilistic forecasting for simple returns."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from core.contracts import (
    ForecastingMethod,
    ForecastInterval,
    ForecastQuantile,
    ProbabilisticForecast,
)

from .simulation import FloatArray, _prepare_returns, _simulate_paths_array, _validate_integer

_DEFAULT_QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
_DEFAULT_INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


def _validate_levels(levels: Sequence[float], name: str) -> tuple[float, ...]:
    """Validate unique, ascending probability levels strictly inside (0, 1)."""
    try:
        checked = tuple(float(value) for value in levels)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric probability levels") from exc
    if not checked:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in checked):
        raise ValueError(f"{name} levels must be strictly between 0 and 1")
    if tuple(sorted(set(checked))) != checked:
        raise ValueError(f"{name} levels must be unique and ascending")
    return checked


def forecast_return_distribution(
    returns: Sequence[float] | FloatArray,
    *,
    horizon: int = 1,
    simulations: int = 10_000,
    random_seed: int = 27,
    interval_levels: Sequence[float] = _DEFAULT_INTERVAL_LEVELS,
    quantile_levels: Sequence[float] = _DEFAULT_QUANTILE_LEVELS,
    regime_labels: Sequence[int] | None = None,
    transition_matrix: Sequence[Sequence[float]] | None = None,
) -> ProbabilisticForecast:
    """Estimate a predictive distribution for compounded cumulative simple returns.

    The empirical bootstrap resamples only the supplied training-window observations.
    Optional F26 regime conditioning requires labels and a transition matrix calibrated
    on that same window; it models future state transitions with a Markov chain and
    samples returns from each state's historical pool.
    """
    checked = _prepare_returns(returns)
    steps = _validate_integer(horizon, "horizon", 1)
    draws = _validate_integer(simulations, "simulations", 100)
    seed = _validate_integer(random_seed, "random_seed", 0)
    intervals = _validate_levels(interval_levels, "interval_levels")
    requested_quantiles = _validate_levels(quantile_levels, "quantile_levels")
    paths, _, labels, matrix = _simulate_paths_array(
        checked,
        horizon=steps,
        simulations=draws,
        random_seed=seed,
        regime_labels=regime_labels,
        transition_matrix=transition_matrix,
    )
    cumulative_returns = np.prod(1.0 + paths, axis=1) - 1.0
    if not np.all(np.isfinite(cumulative_returns)):
        raise RuntimeError("simulated cumulative returns contain non-finite values")

    probabilities = set(requested_quantiles)
    for coverage in intervals:
        probabilities.add((1.0 - coverage) / 2.0)
        probabilities.add((1.0 + coverage) / 2.0)
    sorted_probabilities = tuple(sorted(probabilities))
    quantile_values = np.quantile(cumulative_returns, sorted_probabilities, method="linear")
    quantiles = tuple(
        ForecastQuantile(probability=level, return_value=float(value))
        for level, value in zip(sorted_probabilities, quantile_values, strict=True)
    )
    interval_results = tuple(
        ForecastInterval(
            coverage=coverage,
            lower_return=float(np.quantile(cumulative_returns, (1.0 - coverage) / 2.0)),
            upper_return=float(np.quantile(cumulative_returns, (1.0 + coverage) / 2.0)),
        )
        for coverage in intervals
    )
    method: ForecastingMethod = (
        "markov_regime_bootstrap"
        if labels is not None and matrix is not None
        else ("iid_empirical_bootstrap")
    )
    if method == "iid_empirical_bootstrap":
        methodology = (
            "Predictive distribution from IID empirical bootstrap paths with replacement; "
            "future return dependence is not modelled."
        )
    else:
        methodology = (
            "Predictive distribution from a Markov regime bootstrap using training-only "
            "state labels, a supplied transition matrix, and within-state empirical returns."
        )
    return ProbabilisticForecast(
        model=method,
        training_observations=len(checked),
        horizon=steps,
        simulations=draws,
        random_seed=seed,
        expected_return=float(np.mean(cumulative_returns)),
        median_return=float(np.median(cumulative_returns)),
        predictive_std=float(np.std(cumulative_returns, ddof=1)),
        probability_of_loss=float(np.mean(cumulative_returns < 0.0)),
        quantiles=quantiles,
        intervals=interval_results,
        methodology=methodology,
    )


def forecast_return_distributions(
    returns: Sequence[float] | FloatArray,
    *,
    horizons: Sequence[int] = (1, 5, 10, 20),
    simulations: int = 10_000,
    random_seed: int = 27,
    interval_levels: Sequence[float] = _DEFAULT_INTERVAL_LEVELS,
    quantile_levels: Sequence[float] = _DEFAULT_QUANTILE_LEVELS,
    regime_labels: Sequence[int] | None = None,
    transition_matrix: Sequence[Sequence[float]] | None = None,
) -> tuple[ProbabilisticForecast, ...]:
    """Produce forecasts at several unique horizons with deterministic seed offsets."""
    checked_horizons = tuple(horizons)
    if not checked_horizons:
        raise ValueError("horizons must not be empty")
    for horizon in checked_horizons:
        _validate_integer(horizon, "horizon", 1)
    if len(set(checked_horizons)) != len(checked_horizons):
        raise ValueError("horizons must be unique")
    seed = _validate_integer(random_seed, "random_seed", 0)
    return tuple(
        forecast_return_distribution(
            returns,
            horizon=horizon,
            simulations=simulations,
            random_seed=seed + index,
            interval_levels=interval_levels,
            quantile_levels=quantile_levels,
            regime_labels=regime_labels,
            transition_matrix=transition_matrix,
        )
        for index, horizon in enumerate(checked_horizons)
    )


__all__ = ["forecast_return_distribution", "forecast_return_distributions"]
