"""Forecast calibration scores and strict chronological out-of-sample evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from core.contracts import (
    ForecastCalibrationResult,
    ProbabilisticForecast,
    WalkForwardForecastEvaluation,
)

from .models import forecast_return_distribution
from .simulation import _prepare_returns, _validate_integer

RegimeMetadataProvider = Callable[
    [tuple[float, ...]], tuple[Sequence[int], Sequence[Sequence[float]]]
]


def evaluate_forecast_calibration(
    forecasts: Sequence[ProbabilisticForecast],
    realized_returns: Sequence[float],
    *,
    interval_coverage: float = 0.90,
) -> ForecastCalibrationResult:
    """Score forecasts against aligned realized cumulative returns.

    Reports empirical interval coverage, mean width, the interval score, Brier score
    for the loss event, and point-mean errors. Forecasts and realizations must use the
    same horizon and chronological ordering.
    """
    checked_forecasts = tuple(forecasts)
    try:
        actual = np.asarray(tuple(float(value) for value in realized_returns), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("realized_returns must be numeric") from exc
    if not checked_forecasts:
        raise ValueError("at least one forecast is required")
    if actual.ndim != 1 or actual.size != len(checked_forecasts):
        raise ValueError("realized_returns must align one-to-one with forecasts")
    if not np.all(np.isfinite(actual)) or np.any(actual <= -1.0):
        raise ValueError("realized simple returns must be finite and greater than -1")
    if not math.isfinite(interval_coverage) or not 0.0 < interval_coverage < 1.0:
        raise ValueError("interval_coverage must be strictly between 0 and 1")

    horizon = checked_forecasts[0].horizon
    if any(forecast.horizon != horizon for forecast in checked_forecasts):
        raise ValueError("all forecasts must have the same horizon")
    intervals = []
    for forecast in checked_forecasts:
        matching = tuple(
            interval
            for interval in forecast.intervals
            if math.isclose(interval.coverage, interval_coverage, abs_tol=1e-9)
        )
        if not matching:
            raise ValueError(f"forecast is missing the requested {interval_coverage:.6g} interval")
        intervals.append(matching[0])

    lower = np.asarray([interval.lower_return for interval in intervals], dtype=np.float64)
    upper = np.asarray([interval.upper_return for interval in intervals], dtype=np.float64)
    means = np.asarray([forecast.expected_return for forecast in checked_forecasts])
    loss_probabilities = np.asarray(
        [forecast.probability_of_loss for forecast in checked_forecasts], dtype=np.float64
    )
    covered = (actual >= lower) & (actual <= upper)
    alpha = 1.0 - interval_coverage
    interval_scores = upper - lower
    interval_scores = interval_scores + (2.0 / alpha) * np.maximum(lower - actual, 0.0)
    interval_scores = interval_scores + (2.0 / alpha) * np.maximum(actual - upper, 0.0)
    loss_events = (actual < 0.0).astype(np.float64)
    errors = means - actual

    return ForecastCalibrationResult(
        forecasts=len(checked_forecasts),
        horizon=horizon,
        nominal_coverage=interval_coverage,
        empirical_coverage=float(np.mean(covered)),
        mean_interval_width=float(np.mean(upper - lower)),
        mean_interval_score=float(np.mean(interval_scores)),
        probability_of_loss_brier_score=float(np.mean((loss_probabilities - loss_events) ** 2)),
        mean_absolute_error=float(np.mean(np.abs(errors))),
        mean_squared_error=float(np.mean(errors**2)),
    )


def walk_forward_forecast_evaluation(
    returns: Sequence[float],
    *,
    horizon: int = 1,
    min_train_observations: int = 60,
    step: int = 1,
    simulations: int = 2_000,
    random_seed: int = 27,
    interval_coverage: float = 0.90,
    regime_provider: RegimeMetadataProvider | None = None,
    regime_labels: Sequence[int] | None = None,
    transition_matrix: Sequence[Sequence[float]] | None = None,
) -> WalkForwardForecastEvaluation:
    """Run expanding-window evaluation without fitting on evaluated outcomes.

    At every forecast origin, this function gives a regime provider only the exact
    historical prefix strictly before that origin. The provider must fit/derive both
    labels and transition probabilities from that prefix and return metadata aligned
    to it. Fixed labels or transition matrices are rejected: their estimation window
    cannot be verified here, and a matrix fitted on the complete sample leaks future
    information into earlier forecasts.
    """
    checked = _prepare_returns(returns)
    steps = _validate_integer(horizon, "horizon", 1)
    minimum = _validate_integer(min_train_observations, "min_train_observations", 20)
    stride = _validate_integer(step, "step", 1)
    draws = _validate_integer(simulations, "simulations", 100)
    seed = _validate_integer(random_seed, "random_seed", 0)
    if not math.isfinite(interval_coverage) or not 0.0 < interval_coverage < 1.0:
        raise ValueError("interval_coverage must be strictly between 0 and 1")

    if regime_labels is not None or transition_matrix is not None:
        raise ValueError(
            "walk-forward rejects fixed regime_labels/transition_matrix; "
            "use regime_provider to fit metadata on each training prefix"
        )
    if regime_provider is not None and not callable(regime_provider):
        raise ValueError("regime_provider must be callable")

    origins = tuple(range(minimum, len(checked) - steps + 1, stride))
    if not origins:
        raise ValueError("returns do not contain any complete out-of-sample forecast window")

    forecasts: list[ProbabilisticForecast] = []
    realized: list[float] = []
    for origin in origins:
        train = checked[:origin]
        train_values = tuple(float(value) for value in train)
        train_labels: Sequence[int] | None = None
        train_transition_matrix: Sequence[Sequence[float]] | None = None
        if regime_provider is not None:
            metadata = regime_provider(train_values)
            if not isinstance(metadata, tuple) or len(metadata) != 2:
                raise ValueError(
                    "regime_provider must return (training_regime_labels, transition_matrix)"
                )
            train_labels, train_transition_matrix = metadata
            if len(train_labels) != origin:
                raise ValueError(
                    "regime_provider labels must align exactly with the training prefix"
                )

        forecast = forecast_return_distribution(
            train,
            horizon=steps,
            simulations=draws,
            random_seed=seed + origin,
            interval_levels=(interval_coverage,),
            regime_labels=train_labels,
            transition_matrix=train_transition_matrix,
        )
        test_window = checked[origin : origin + steps]
        realized.append(float(np.prod(1.0 + test_window) - 1.0))
        forecasts.append(forecast)

    calibration = evaluate_forecast_calibration(
        forecasts,
        realized,
        interval_coverage=interval_coverage,
    )
    return WalkForwardForecastEvaluation(
        horizon=steps,
        min_train_observations=minimum,
        step=stride,
        origins=origins,
        forecasts=tuple(forecasts),
        realized_returns=tuple(realized),
        calibration=calibration,
    )


__all__ = ["evaluate_forecast_calibration", "walk_forward_forecast_evaluation"]
