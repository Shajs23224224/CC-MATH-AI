"""Numerical, contract and no-look-ahead tests for F27 forecasting."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from core.contracts import (
    ForecastInterval,
    ForecastQuantile,
    ProbabilisticForecast,
)
from quant.forecasting import (
    evaluate_forecast_calibration,
    forecast_return_distribution,
    forecast_return_distributions,
    simulate_return_paths,
    walk_forward_forecast_evaluation,
)


def _sample_returns(size: int = 180) -> tuple[float, ...]:
    rng = np.random.default_rng(2701)
    return tuple(float(value) for value in rng.normal(0.0008, 0.018, size=size))


def _manual_forecast(
    *,
    expected_return: float,
    probability_of_loss: float,
    lower: float,
    upper: float,
) -> ProbabilisticForecast:
    return ProbabilisticForecast(
        model="iid_empirical_bootstrap",
        training_observations=100,
        horizon=1,
        simulations=500,
        random_seed=27,
        expected_return=expected_return,
        median_return=expected_return,
        predictive_std=0.03,
        probability_of_loss=probability_of_loss,
        quantiles=(
            ForecastQuantile(probability=0.05, return_value=lower),
            ForecastQuantile(probability=0.5, return_value=expected_return),
            ForecastQuantile(probability=0.95, return_value=upper),
        ),
        intervals=(ForecastInterval(coverage=0.9, lower_return=lower, upper_return=upper),),
        methodology="Test fixture.",
    )


def test_bootstrap_forecast_is_reproducible_and_has_ordered_intervals() -> None:
    returns = _sample_returns()
    first = forecast_return_distribution(returns, horizon=5, simulations=1_000, random_seed=2702)
    second = forecast_return_distribution(returns, horizon=5, simulations=1_000, random_seed=2702)

    assert first == second
    assert first.model == "iid_empirical_bootstrap"
    assert first.training_observations == len(returns)
    assert first.horizon == 5
    assert np.isfinite(first.expected_return) and first.expected_return > -1.0
    assert 0.0 <= first.probability_of_loss <= 1.0
    assert first.predictive_std > 0.0
    assert [item.probability for item in first.quantiles] == sorted(
        item.probability for item in first.quantiles
    )
    assert [item.return_value for item in first.quantiles] == sorted(
        item.return_value for item in first.quantiles
    )
    intervals = {item.coverage: item for item in first.intervals}
    assert intervals[0.9].lower_return < intervals[0.9].upper_return
    assert intervals[0.95].lower_return <= intervals[0.9].lower_return
    assert intervals[0.95].upper_return >= intervals[0.9].upper_return


def test_multiple_horizons_are_returned_in_requested_order() -> None:
    forecasts = forecast_return_distributions(
        _sample_returns(),
        horizons=(1, 5, 10, 20),
        simulations=500,
        random_seed=2703,
    )

    assert tuple(forecast.horizon for forecast in forecasts) == (1, 5, 10, 20)
    assert len({forecast.random_seed for forecast in forecasts}) == len(forecasts)
    assert all(forecast.training_observations == 180 for forecast in forecasts)


def test_return_path_simulation_is_seeded_and_records_methodology() -> None:
    returns = _sample_returns()
    first = simulate_return_paths(returns, horizon=4, simulations=300, random_seed=2704)
    second = simulate_return_paths(returns, horizon=4, simulations=300, random_seed=2704)

    assert first == second
    assert first.model == "iid_empirical_bootstrap"
    assert first.simulations == 300
    assert len(first.simulated_returns) == 300
    assert all(len(path) == 4 for path in first.simulated_returns)
    assert all(value > -1.0 for path in first.simulated_returns for value in path)


def test_markov_regime_bootstrap_uses_transition_matrix_and_state_pools() -> None:
    rng = np.random.default_rng(2705)
    returns = tuple(
        float(value)
        for value in np.concatenate((rng.normal(-0.02, 0.006, 40), rng.normal(0.025, 0.008, 40)))
    )
    labels = (0,) * 40 + (1,) * 40
    transitions = ((0.90, 0.10), (0.15, 0.85))

    simulation = simulate_return_paths(
        returns,
        horizon=5,
        simulations=500,
        random_seed=2706,
        regime_labels=labels,
        transition_matrix=transitions,
    )
    forecast = forecast_return_distribution(
        returns,
        horizon=5,
        simulations=1_000,
        random_seed=2706,
        regime_labels=labels,
        transition_matrix=transitions,
    )

    assert simulation.model == "markov_regime_bootstrap"
    assert simulation.current_regime == 1
    assert simulation.state_count == 2
    assert simulation.transition_matrix == transitions
    assert forecast.model == "markov_regime_bootstrap"
    assert forecast.expected_return > 0.0


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        ((0.01,) * 19, {}, "at least 20"),
        ((0.01,) * 20 + (float("nan"),), {}, "finite"),
        ((-1.0,) + (0.01,) * 30, {}, "greater than -1"),
        ((0.01,) * 40, {}, "positive sample variance"),
        (_sample_returns(), {"horizon": 0}, "horizon"),
        (_sample_returns(), {"simulations": 99}, "simulations"),
        (_sample_returns(), {"random_seed": -1}, "random_seed"),
        (_sample_returns(), {"interval_levels": (0.9, 0.8)}, "unique and ascending"),
    ],
)
def test_invalid_forecast_inputs_are_rejected(
    values: tuple[float, ...],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        forecast_return_distribution(values, **kwargs)  # type: ignore[arg-type]


def test_regime_inputs_require_labels_matrix_alignment_and_stochastic_rows() -> None:
    returns = _sample_returns(60)
    labels = (0,) * 30 + (1,) * 30
    with pytest.raises(ValueError, match="supplied together"):
        simulate_return_paths(returns, horizon=2, simulations=100, regime_labels=labels)
    with pytest.raises(ValueError, match="align exactly"):
        simulate_return_paths(
            returns[:-1],
            horizon=2,
            simulations=100,
            regime_labels=labels,
            transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        )
    with pytest.raises(ValueError, match="sum to one"):
        forecast_return_distribution(
            returns,
            horizon=2,
            simulations=100,
            regime_labels=labels,
            transition_matrix=((0.9, 0.2), (0.2, 0.8)),
        )


def test_calibration_metrics_match_small_hand_calculated_example() -> None:
    forecasts = (
        _manual_forecast(expected_return=0.01, probability_of_loss=0.2, lower=-0.10, upper=0.10),
        _manual_forecast(expected_return=-0.02, probability_of_loss=0.7, lower=-0.05, upper=0.05),
    )

    result = evaluate_forecast_calibration(forecasts, (0.0, 0.06), interval_coverage=0.9)

    assert result.forecasts == 2
    assert result.horizon == 1
    assert result.empirical_coverage == pytest.approx(0.5)
    assert result.mean_interval_width == pytest.approx(0.15)
    assert result.mean_interval_score == pytest.approx(0.25)
    assert result.probability_of_loss_brier_score == pytest.approx(0.265)
    assert result.mean_absolute_error == pytest.approx(0.045)
    assert result.mean_squared_error == pytest.approx(0.00325)


def test_calibration_rejects_misaligned_realizations_and_missing_intervals() -> None:
    forecast = _manual_forecast(
        expected_return=0.01, probability_of_loss=0.2, lower=-0.10, upper=0.10
    )
    with pytest.raises(ValueError, match="align one-to-one"):
        evaluate_forecast_calibration((forecast,), (0.0, 0.01))
    with pytest.raises(ValueError, match="missing the requested"):
        evaluate_forecast_calibration((forecast,), (0.0,), interval_coverage=0.8)


def test_walk_forward_evaluation_trains_only_before_each_origin() -> None:
    returns = _sample_returns(85)
    altered_future = list(returns)
    altered_future[60:] = [value + 0.08 for value in altered_future[60:]]

    original = walk_forward_forecast_evaluation(
        returns,
        horizon=1,
        min_train_observations=40,
        step=5,
        simulations=200,
        random_seed=2707,
    )
    altered = walk_forward_forecast_evaluation(
        tuple(altered_future),
        horizon=1,
        min_train_observations=40,
        step=5,
        simulations=200,
        random_seed=2707,
    )

    original_index = original.origins.index(60)
    altered_index = altered.origins.index(60)
    assert original.forecasts[original_index] == altered.forecasts[altered_index]
    assert original.forecasts[original_index].training_observations == 60
    assert original.realized_returns[original_index] != altered.realized_returns[altered_index]
    assert original.calibration.forecasts == len(original.forecasts)
    assert original.calibration.horizon == 1
    assert all(
        forecast.training_observations == origin
        for origin, forecast in zip(original.origins, original.forecasts, strict=True)
    )


def test_walk_forward_rejects_insufficient_out_of_sample_window() -> None:
    with pytest.raises(ValueError, match="any complete out-of-sample"):
        walk_forward_forecast_evaluation(_sample_returns(40), min_train_observations=40, horizon=1)


def test_forecast_contract_rejects_reversed_interval_and_unsorted_quantiles() -> None:
    with pytest.raises(ValidationError, match="lower_return"):
        ForecastInterval(coverage=0.9, lower_return=0.1, upper_return=-0.1)
    with pytest.raises(ValidationError, match="unique and ascending"):
        ProbabilisticForecast(
            model="iid_empirical_bootstrap",
            training_observations=100,
            horizon=1,
            simulations=500,
            random_seed=27,
            expected_return=0.0,
            median_return=0.0,
            predictive_std=0.02,
            probability_of_loss=0.5,
            quantiles=(
                ForecastQuantile(probability=0.5, return_value=0.0),
                ForecastQuantile(probability=0.1, return_value=-0.03),
            ),
            intervals=(ForecastInterval(coverage=0.9, lower_return=-0.1, upper_return=0.1),),
            methodology="invalid fixture",
        )
