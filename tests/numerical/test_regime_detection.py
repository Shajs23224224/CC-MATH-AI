"""Numerical and safety tests for F26 regime detection."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from core.contracts import RegimeModelResult, VolatilityRegimeThresholds
from quant.regimes import (
    apply_volatility_regimes,
    bayesian_change_point_probabilities,
    cusum_change_points,
    filter_regime_observations,
    fit_gaussian_hmm,
    fit_markov_switching_gaussian,
    fit_volatility_regime_thresholds,
)


def _two_regime_series() -> np.ndarray:
    rng = np.random.default_rng(2601)
    low = rng.normal(-1.0, 0.18, size=180)
    high = rng.normal(1.2, 0.24, size=180)
    return np.concatenate((low, high))


def test_gaussian_hmm_recovers_two_separated_means_and_probability_simplices() -> None:
    observations = _two_regime_series()

    result = fit_gaussian_hmm(observations, n_states=2, max_iterations=100)

    filtered = np.asarray(result.filtered_probabilities)
    transition = np.asarray(result.transition_matrix)
    assert result.model == "gaussian_hmm"
    assert result.observations == 360
    assert result.means[0] < result.means[1]
    assert result.means[1] - result.means[0] > 1.5
    assert np.allclose(filtered.sum(axis=1), 1.0)
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.all(np.isfinite(filtered))
    assert np.mean(np.asarray(result.filtered_states)[-100:] == 1) > 0.90
    assert result.iterations >= 1
    assert np.isfinite(result.log_likelihood)


def test_hmm_rejects_constant_and_non_finite_observations() -> None:
    with pytest.raises(ValueError, match="positive sample variance"):
        fit_gaussian_hmm((1.0,) * 40)
    with pytest.raises(ValueError, match="finite values"):
        fit_gaussian_hmm((0.0,) * 30 + (float("nan"),) + (1.0,) * 20)
    with pytest.raises(ValueError, match="n_states"):
        fit_gaussian_hmm(_two_regime_series(), n_states=1)


def test_markov_switching_model_uses_the_same_auditable_state_contract() -> None:
    result = fit_markov_switching_gaussian(_two_regime_series(), n_states=2)

    assert result.model == "markov_switching_gaussian"
    assert len(result.transition_matrix) == 2
    assert len(result.filtered_probabilities) == result.observations
    assert "first-order latent Markov state" in result.methodology


def test_cusum_detects_upward_shift_online() -> None:
    observations = tuple([0.0] * 30 + [2.5] * 12)

    result = cusum_change_points(
        observations,
        reference_mean=0.0,
        reference_std=1.0,
        drift=0.25,
        threshold=4.0,
    )

    upward = tuple(point for point in result.change_points if point.direction == "upward")
    assert upward
    assert upward[0].index >= 30
    assert len(result.upper_cusum) == len(observations)
    assert max(result.upper_cusum) >= 4.0


def test_cusum_rejects_invalid_calibration_parameters() -> None:
    observations = tuple(np.random.default_rng(26).normal(size=40))
    with pytest.raises(ValueError, match="reference_std"):
        cusum_change_points(observations, reference_mean=0.0, reference_std=0.0)
    with pytest.raises(ValueError, match="threshold"):
        cusum_change_points(observations, reference_mean=0.0, reference_std=1.0, threshold=0.0)


def test_bayesian_change_point_posterior_is_bounded_and_reacts_to_shift() -> None:
    rng = np.random.default_rng(2602)
    observations = np.concatenate((rng.normal(0.0, 0.25, size=40), rng.normal(2.0, 0.25, size=30)))

    result = bayesian_change_point_probabilities(
        observations,
        hazard=0.02,
        prior_mean=0.0,
        prior_kappa=1.0,
        prior_alpha=2.0,
        prior_beta=0.25,
        max_run_length=100,
    )

    probabilities = np.asarray(result.change_probabilities)
    assert result.observations == len(observations)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert len(result.most_likely_run_lengths) == len(observations)
    assert np.max(probabilities[40:47]) > np.median(probabilities[:30])
    assert all(run < result.max_run_length for run in result.most_likely_run_lengths)


def test_bayesian_change_point_rejects_invalid_hazard_and_run_limit() -> None:
    observations = tuple(np.random.default_rng(5).normal(size=30))
    with pytest.raises(ValueError, match="hazard"):
        bayesian_change_point_probabilities(observations, hazard=1.0)
    with pytest.raises(ValueError, match="max_run_length"):
        bayesian_change_point_probabilities(observations, max_run_length=1)


def test_volatility_regimes_use_training_calibrated_thresholds() -> None:
    rng = np.random.default_rng(2603)
    calibration = np.concatenate(
        (rng.normal(0.0, 0.008, size=150), rng.normal(0.0, 0.05, size=150))
    )
    thresholds = fit_volatility_regime_thresholds(
        calibration,
        window=20,
        low_quantile=0.33,
        high_quantile=0.67,
        version="training-window-v1",
    )
    target = np.concatenate((rng.normal(0.0, 0.004, size=40), rng.normal(0.0, 0.07, size=40)))
    result = apply_volatility_regimes(target, thresholds)

    assert result.version == "training-window-v1"
    assert result.thresholds == thresholds
    assert len(result.labels) == len(target) - thresholds.window + 1
    assert result.labels[:10].count("low") >= 5
    assert result.labels[-10:].count("high") >= 5
    assert result.window_end_indices[0] == 20
    assert result.window_end_indices[-1] == len(target)


def test_volatility_thresholds_reject_indistinguishable_training_quantiles() -> None:
    with pytest.raises(ValueError, match="distinct"):
        fit_volatility_regime_thresholds((0.01,) * 100, window=10)
    with pytest.raises(ValueError, match="quantiles"):
        fit_volatility_regime_thresholds(
            tuple(np.random.default_rng(12).normal(size=100)),
            window=10,
            low_quantile=0.8,
            high_quantile=0.4,
        )


def test_regime_contract_rejects_invalid_probability_matrix() -> None:
    with pytest.raises(ValidationError, match="transition matrix row must sum"):
        RegimeModelResult(
            model="gaussian_hmm",
            version="test-v1",
            observations=20,
            state_count=2,
            means=(-1.0, 1.0),
            variances=(0.2, 0.2),
            initial_probabilities=(0.5, 0.5),
            transition_matrix=((0.9, 0.2), (0.1, 0.9)),
            filtered_probabilities=((0.5, 0.5),) * 20,
            smoothed_probabilities=((0.5, 0.5),) * 20,
            filtered_states=(0,) * 20,
            smoothed_states=(0,) * 20,
            log_likelihood=-20.0,
            converged=True,
            iterations=10,
            methodology="invalid contract fixture",
        )


def test_volatility_regime_contract_requires_threshold_window_match() -> None:
    thresholds = VolatilityRegimeThresholds(
        version="v1",
        window=5,
        calibration_observations=10,
        low_quantile=0.25,
        high_quantile=0.75,
        low_threshold=0.1,
        high_threshold=0.2,
    )
    with pytest.raises(ValidationError, match="same window length"):
        from core.contracts import VolatilityRegimeSeries

        VolatilityRegimeSeries(
            version="v1",
            observations=6,
            window=3,
            window_end_indices=(3, 4, 5, 6),
            realized_volatility=(0.05, 0.15, 0.25, 0.1),
            labels=("low", "normal", "high", "normal"),
            thresholds=thresholds,
            methodology="invalid contract fixture",
        )


def test_held_out_filter_uses_fixed_training_parameters_and_is_causal() -> None:
    rng = np.random.default_rng(2604)
    training = np.concatenate((rng.normal(-0.8, 0.2, 120), rng.normal(0.9, 0.25, 120)))
    fitted = fit_gaussian_hmm(training, n_states=2, max_iterations=80)
    held_out = np.concatenate((rng.normal(-0.8, 0.2, 30), rng.normal(1.0, 0.25, 30)))

    first = filter_regime_observations(held_out[:40], fitted)
    full = filter_regime_observations(held_out, fitted)

    assert first.training_observations == len(training)
    assert first.observations == 40
    assert full.observations == len(held_out)
    assert np.allclose(first.filtered_probabilities, full.filtered_probabilities[:40], atol=1e-10)
    assert np.allclose(np.asarray(full.filtered_probabilities).sum(axis=1), 1.0)
    assert "fixed parameters" in full.methodology
