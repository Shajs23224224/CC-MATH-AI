"""Latent-state regime models, change-point detection and volatility labels for F26."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from core.contracts.regimes import (
    BayesianChangePointResult,
    ChangePoint,
    CUSUMResult,
    RegimeModel,
    RegimeModelResult,
    RegimeProbabilitySeries,
    VolatilityRegimeSeries,
    VolatilityRegimeThresholds,
)

_EPSILON = 1e-12
_RegimeMethod = Literal["gaussian_hmm", "markov_switching_gaussian"]


def _validate_observations(
    values: Sequence[float],
    *,
    name: str = "observations",
    minimum: int = 20,
) -> np.ndarray:
    """Return a finite one-dimensional sample in the original observation order."""
    try:
        checked = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from exc
    if checked.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if checked.size < minimum:
        raise ValueError(f"{name} requires at least {minimum} observations")
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain only finite values; missing data is not imputed")
    return checked


def _log_emissions(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Compute Gaussian log densities for every observation and latent state."""
    return -0.5 * (
        math.log(2.0 * math.pi) + np.log(variances)[None, :]
        + (values[:, None] - means[None, :]) ** 2 / variances[None, :]
    )


def _forward_backward(
    log_emissions: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute likelihood, smoothed/filtered probabilities and transition counts."""
    log_initial = np.log(np.maximum(initial, _EPSILON))
    safe_transition = np.maximum(transition, _EPSILON)
    safe_transition /= safe_transition.sum(axis=1, keepdims=True)
    log_transition = np.log(safe_transition)
    observation_count, state_count = log_emissions.shape

    log_alpha = np.empty((observation_count, state_count), dtype=np.float64)
    log_alpha[0] = log_initial + log_emissions[0]
    for index in range(1, observation_count):
        log_alpha[index] = log_emissions[index] + logsumexp(
            log_alpha[index - 1][:, None] + log_transition,
            axis=0,
        )

    log_likelihood = float(logsumexp(log_alpha[-1]))
    log_beta = np.zeros((observation_count, state_count), dtype=np.float64)
    for index in range(observation_count - 2, -1, -1):
        log_beta[index] = logsumexp(
            log_transition
            + log_emissions[index + 1][None, :]
            + log_beta[index + 1][None, :],
            axis=1,
        )

    log_smoothed = log_alpha + log_beta - log_likelihood
    smoothed = np.exp(log_smoothed)
    smoothed /= smoothed.sum(axis=1, keepdims=True)

    log_filtered = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    filtered = np.exp(log_filtered)

    transition_counts = np.zeros((state_count, state_count), dtype=np.float64)
    for index in range(observation_count - 1):
        log_xi = (
            log_alpha[index][:, None]
            + log_transition
            + log_emissions[index + 1][None, :]
            + log_beta[index + 1][None, :]
            - log_likelihood
        )
        xi = np.exp(log_xi)
        transition_counts += xi / max(float(xi.sum()), _EPSILON)

    return log_likelihood, filtered, smoothed, transition_counts


def _fit_gaussian_markov_model(
    observations: Sequence[float],
    *,
    model: _RegimeMethod,
    n_states: int,
    max_iterations: int,
    tolerance: float,
) -> RegimeModelResult:
    """Fit Gaussian state emissions and first-order Markov transitions by EM."""
    values = _validate_observations(observations, minimum=max(20, n_states * 5))
    if not isinstance(n_states, int) or isinstance(n_states, bool) or not 2 <= n_states <= 6:
        raise ValueError("n_states must be an integer between 2 and 6")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    global_variance = float(np.var(values, ddof=1))
    if not math.isfinite(global_variance) or global_variance <= _EPSILON:
        raise ValueError("regime fitting requires positive sample variance")

    quantiles = np.linspace(0.1, 0.9, n_states)
    means = np.asarray(np.quantile(values, quantiles), dtype=np.float64)
    variances = np.full(n_states, global_variance, dtype=np.float64)
    initial = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    transition = np.full(
        (n_states, n_states),
        0.15 / max(n_states - 1, 1),
        dtype=np.float64,
    )
    np.fill_diagonal(transition, 0.85)
    variance_floor = max(global_variance * 1e-6, 1e-12)
    previous_likelihood: float | None = None
    converged = False
    iterations = max_iterations

    for iteration in range(1, max_iterations + 1):
        emissions = _log_emissions(values, means, variances)
        likelihood, _, smoothed, transition_counts = _forward_backward(
            emissions, initial, transition
        )
        if previous_likelihood is not None:
            improvement = abs(likelihood - previous_likelihood)
            if improvement <= tolerance * (1.0 + abs(previous_likelihood)):
                converged = True
                iterations = iteration
                break

        previous_likelihood = likelihood
        state_mass = np.maximum(smoothed.sum(axis=0), _EPSILON)
        new_means = (smoothed.T @ values) / state_mass
        new_variances = np.empty(n_states, dtype=np.float64)
        for state in range(n_states):
            residuals = values - new_means[state]
            new_variances[state] = max(
                float(np.sum(smoothed[:, state] * residuals * residuals) / state_mass[state]),
                variance_floor,
            )

        new_initial = np.maximum(smoothed[0], _EPSILON)
        new_initial /= new_initial.sum()
        new_transition = transition_counts + 1e-8
        new_transition /= new_transition.sum(axis=1, keepdims=True)

        # Canonical state ordering makes runs reproducible and regime indices interpretable.
        order = np.argsort(new_means, kind="stable")
        means = new_means[order]
        variances = new_variances[order]
        initial = new_initial[order]
        transition = new_transition[np.ix_(order, order)]

    emissions = _log_emissions(values, means, variances)
    likelihood, filtered, smoothed, _ = _forward_backward(emissions, initial, transition)
    filtered_states = np.argmax(filtered, axis=1)
    smoothed_states = np.argmax(smoothed, axis=1)

    if not math.isfinite(likelihood):
        raise ValueError("regime model produced a non-finite log likelihood")

    methodology = (
        "Gaussian emissions with state-specific means and variances, first-order Markov "
        "transition matrix and deterministic EM initialization by observation quantiles. "
        "Filtered probabilities use observations available up to each index; smoothed "
        "probabilities condition on the full sample and therefore are not live signals."
    )
    if model == "markov_switching_gaussian":
        methodology = (
            "Gaussian Markov-switching distribution fitted by EM: both mean and variance "
            "switch by a first-order latent Markov state. The state-space formulation is "
            "equivalent to a Gaussian-emission hidden Markov model. " + methodology
        )

    return RegimeModelResult(
        model=model,
        version="gaussian-regime-em-v1",
        observations=int(values.size),
        state_count=n_states,
        means=tuple(float(value) for value in means),
        variances=tuple(float(value) for value in variances),
        initial_probabilities=tuple(float(value) for value in initial),
        transition_matrix=tuple(
            tuple(float(value) for value in row) for row in transition
        ),
        filtered_probabilities=tuple(
            tuple(float(value) for value in row) for row in filtered
        ),
        smoothed_probabilities=tuple(
            tuple(float(value) for value in row) for row in smoothed
        ),
        filtered_states=tuple(int(value) for value in filtered_states),
        smoothed_states=tuple(int(value) for value in smoothed_states),
        log_likelihood=likelihood,
        converged=converged,
        iterations=iterations,
        methodology=methodology,
    )


def fit_gaussian_hmm(
    observations: Sequence[float],
    *,
    n_states: int = 2,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> RegimeModelResult:
    """Fit a univariate Gaussian hidden Markov model with deterministic EM."""
    return _fit_gaussian_markov_model(
        observations,
        model="gaussian_hmm",
        n_states=n_states,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def fit_markov_switching_gaussian(
    observations: Sequence[float],
    *,
    n_states: int = 2,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> RegimeModelResult:
    """Fit a Gaussian Markov-switching mean/variance model.

    The model is represented as a first-order latent Markov chain with
    state-dependent Gaussian emissions, allowing a single stable likelihood and
    probability contract for the two common regime-model names.
    """
    return _fit_gaussian_markov_model(
        observations,
        model="markov_switching_gaussian",
        n_states=n_states,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def cusum_change_points(
    observations: Sequence[float],
    *,
    reference_mean: float,
    reference_std: float,
    drift: float = 0.5,
    threshold: float = 5.0,
) -> CUSUMResult:
    """Detect sequential mean shifts using a calibrated two-sided CUSUM.

    Reference parameters should be estimated on a training/calibration window.
    The detector processes each observation once and never uses future values to
    decide whether a current index crosses the threshold.
    """
    values = _validate_observations(observations)
    if not math.isfinite(reference_mean):
        raise ValueError("reference_mean must be finite")
    if not math.isfinite(reference_std) or reference_std <= 0.0:
        raise ValueError("reference_std must be finite and positive")
    if not math.isfinite(drift) or drift < 0.0:
        raise ValueError("drift must be finite and non-negative")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")

    upper = 0.0
    lower = 0.0
    upper_path: list[float] = []
    lower_path: list[float] = []
    changes: list[ChangePoint] = []
    for index, value in enumerate(values):
        standardized = (float(value) - reference_mean) / reference_std
        upper = max(0.0, upper + standardized - drift)
        lower = max(0.0, lower - standardized - drift)
        upper_path.append(upper)
        lower_path.append(lower)
        if upper >= threshold:
            changes.append(ChangePoint(index=index, direction="upward", score=float(upper)))
            upper = 0.0
            lower = 0.0
        elif lower >= threshold:
            changes.append(ChangePoint(index=index, direction="downward", score=float(lower)))
            upper = 0.0
            lower = 0.0

    return CUSUMResult(
        observations=int(values.size),
        reference_mean=float(reference_mean),
        reference_std=float(reference_std),
        drift=float(drift),
        threshold=float(threshold),
        upper_cusum=tuple(upper_path),
        lower_cusum=tuple(lower_path),
        change_points=tuple(changes),
        methodology=(
            "Sequential standardized two-sided CUSUM with caller-calibrated reference mean, "
            "standard deviation, allowance (drift) and alarm threshold; both accumulators "
            "reset after an alarm."
        ),
    )


def _update_segment_stats(
    count: int,
    mean: float,
    sum_squared_deviations: float,
    value: float,
) -> tuple[int, float, float]:
    """Update online segment count, mean and centered sum of squares."""
    next_count = count + 1
    delta = value - mean
    next_mean = mean + delta / next_count
    next_m2 = sum_squared_deviations + delta * (value - next_mean)
    return next_count, next_mean, next_m2


def _normal_inverse_gamma_log_predictive(
    value: float,
    stats: tuple[int, float, float],
    *,
    prior_mean: float,
    prior_kappa: float,
    prior_alpha: float,
    prior_beta: float,
) -> float:
    """Student-t log predictive density under a Normal-Inverse-Gamma prior."""
    count, sample_mean, sum_squared_deviations = stats
    posterior_kappa = prior_kappa + count
    posterior_mean = (
        prior_kappa * prior_mean + count * sample_mean
    ) / posterior_kappa
    posterior_alpha = prior_alpha + count / 2.0
    posterior_beta = prior_beta + 0.5 * sum_squared_deviations
    if count > 0:
        posterior_beta += (
            prior_kappa
            * count
            * (sample_mean - prior_mean) ** 2
            / (2.0 * posterior_kappa)
        )
    degrees_of_freedom = 2.0 * posterior_alpha
    scale = math.sqrt(
        posterior_beta * (posterior_kappa + 1.0)
        / (posterior_alpha * posterior_kappa)
    )
    return float(
        student_t.logpdf(value, df=degrees_of_freedom, loc=posterior_mean, scale=scale)
    )


def bayesian_change_point_probabilities(
    observations: Sequence[float],
    *,
    hazard: float = 0.01,
    prior_mean: float = 0.0,
    prior_kappa: float = 1.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    max_run_length: int = 500,
) -> BayesianChangePointResult:
    """Return online Bayesian change-point probabilities with an NIG predictive model.

    At each time the posterior probability of a new run is computed using a
    constant hazard and the prior predictive density. The run-length distribution
    is truncated at max_run_length to cap memory and compute time.
    """
    values = _validate_observations(observations)
    if not math.isfinite(hazard) or not 0.0 < hazard < 1.0:
        raise ValueError("hazard must be finite and strictly between zero and one")
    prior_parameters = (prior_mean, prior_kappa, prior_alpha, prior_beta)
    if not all(math.isfinite(value) for value in prior_parameters):
        raise ValueError("prior parameters must be finite")
    if prior_kappa <= 0.0 or prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("prior_kappa, prior_alpha and prior_beta must be positive")
    if (
        not isinstance(max_run_length, int)
        or isinstance(max_run_length, bool)
        or max_run_length < 2
    ):
        raise ValueError("max_run_length must be an integer of at least two")

    prior_stats = (0, float(prior_mean), 0.0)
    run_stats = [prior_stats]
    run_probabilities = np.asarray([1.0], dtype=np.float64)
    change_probabilities: list[float] = []
    most_likely_run_lengths: list[int] = []
    log_hazard = math.log(hazard)
    log_survival = math.log1p(-hazard)

    for value in values:
        log_predictive = np.asarray(
            [
                _normal_inverse_gamma_log_predictive(
                    float(value),
                    stats,
                    prior_mean=prior_mean,
                    prior_kappa=prior_kappa,
                    prior_alpha=prior_alpha,
                    prior_beta=prior_beta,
                )
                for stats in run_stats
            ],
            dtype=np.float64,
        )
        reset_log_mass = log_hazard + _normal_inverse_gamma_log_predictive(
            float(value),
            prior_stats,
            prior_mean=prior_mean,
            prior_kappa=prior_kappa,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )
        growth_log_mass = (
            np.log(np.maximum(run_probabilities, _EPSILON))
            + log_survival
            + log_predictive
        )
        next_log_masses = np.concatenate(([reset_log_mass], growth_log_mass))
        if next_log_masses.size > max_run_length:
            next_log_masses = next_log_masses[:max_run_length]
        normalizer = float(logsumexp(next_log_masses))
        next_probabilities = np.exp(next_log_masses - normalizer)
        next_probabilities /= next_probabilities.sum()

        change_probabilities.append(float(next_probabilities[0]))
        most_likely_run_lengths.append(int(np.argmax(next_probabilities)))

        next_stats = [_update_segment_stats(*prior_stats, float(value))]
        next_stats.extend(
            _update_segment_stats(*stats, float(value))
            for stats in run_stats
        )
        run_stats = next_stats[:max_run_length]
        run_probabilities = next_probabilities

    return BayesianChangePointResult(
        observations=int(values.size),
        hazard=float(hazard),
        max_run_length=max_run_length,
        change_probabilities=tuple(change_probabilities),
        most_likely_run_lengths=tuple(most_likely_run_lengths),
        prior_mean=float(prior_mean),
        prior_kappa=float(prior_kappa),
        prior_alpha=float(prior_alpha),
        prior_beta=float(prior_beta),
        methodology=(
            "Online Bayesian run-length recursion with constant hazard and a conjugate "
            "Normal-Inverse-Gamma prior; predictive likelihoods use Student-t distributions. "
            "Run-length states are truncated at max_run_length."
        ),
    )


def _rolling_standard_deviation(values: np.ndarray, window: int) -> np.ndarray:
    """Compute sample standard deviation for each trailing window."""
    return np.asarray(
        [
            float(np.std(values[end - window : end], ddof=1))
            for end in range(window, values.size + 1)
        ],
        dtype=np.float64,
    )


def fit_volatility_regime_thresholds(
    calibration_returns: Sequence[float],
    *,
    window: int = 20,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
    version: str = "rolling-vol-quantiles-v1",
) -> VolatilityRegimeThresholds:
    """Fit low/high realized-volatility quantiles using calibration data only."""
    values = _validate_observations(calibration_returns, minimum=window + 9, name="calibration_returns")
    if not isinstance(window, int) or isinstance(window, bool) or window < 3:
        raise ValueError("window must be an integer of at least three")
    if window > values.size - 9:
        raise ValueError("calibration_returns must provide at least ten rolling volatility values")
    if (
        not math.isfinite(low_quantile)
        or not math.isfinite(high_quantile)
        or not 0.0 < low_quantile < high_quantile < 1.0
    ):
        raise ValueError("quantiles must satisfy 0 < low_quantile < high_quantile < 1")

    volatility_values = _rolling_standard_deviation(values, window)
    low_threshold = float(np.quantile(volatility_values, low_quantile))
    high_threshold = float(np.quantile(volatility_values, high_quantile))
    if not math.isfinite(low_threshold) or not math.isfinite(high_threshold):
        raise ValueError("calibrated volatility thresholds must be finite")
    if low_threshold >= high_threshold:
        raise ValueError("calibration does not produce distinct low/high volatility thresholds")
    return VolatilityRegimeThresholds(
        version=version,
        window=window,
        calibration_observations=int(volatility_values.size),
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def apply_volatility_regimes(
    returns: Sequence[float],
    thresholds: VolatilityRegimeThresholds,
) -> VolatilityRegimeSeries:
    """Apply fixed training-calibrated thresholds to trailing realized volatility."""
    values = _validate_observations(returns, minimum=thresholds.window, name="returns")
    volatility_values = _rolling_standard_deviation(values, thresholds.window)
    labels = tuple(
        "low" if value < thresholds.low_threshold
        else "high" if value > thresholds.high_threshold
        else "normal"
        for value in volatility_values
    )
    return VolatilityRegimeSeries(
        version=thresholds.version,
        observations=int(values.size),
        window=thresholds.window,
        window_end_indices=tuple(range(thresholds.window, values.size + 1)),
        realized_volatility=tuple(float(value) for value in volatility_values),
        labels=labels,
        thresholds=thresholds,
        methodology=(
            "Trailing sample standard deviation compared with fixed low/high quantile "
            "thresholds calibrated on a separate training window; no future observations "
            "are used to fit thresholds."
        ),
    )


def filter_regime_observations(
    observations: Sequence[float],
    fitted_model: RegimeModelResult,
) -> RegimeProbabilitySeries:
    """Filter a new chronological sample using parameters fitted on a prior training window.

    The model parameters are held fixed. Each probability at time t conditions only on
    the new observations through t; no parameter refit or backward smoothing is performed.
    """
    values = _validate_observations(observations, name="observations")
    state_count = fitted_model.state_count
    means = np.asarray(fitted_model.means, dtype=np.float64)
    variances = np.asarray(fitted_model.variances, dtype=np.float64)
    initial = np.asarray(fitted_model.initial_probabilities, dtype=np.float64)
    transition = np.asarray(fitted_model.transition_matrix, dtype=np.float64)
    log_transition = np.log(np.maximum(transition, _EPSILON))
    log_emissions = _log_emissions(values, means, variances)

    log_alpha = np.empty((values.size, state_count), dtype=np.float64)
    filtered = np.empty((values.size, state_count), dtype=np.float64)
    log_likelihood = 0.0
    current = np.log(np.maximum(initial, _EPSILON)) + log_emissions[0]
    scale = float(logsumexp(current))
    current -= scale
    log_likelihood += scale
    log_alpha[0] = current
    filtered[0] = np.exp(current)

    for index in range(1, values.size):
        predicted = logsumexp(log_alpha[index - 1][:, None] + log_transition, axis=0)
        current = predicted + log_emissions[index]
        scale = float(logsumexp(current))
        current -= scale
        log_likelihood += scale
        log_alpha[index] = current
        filtered[index] = np.exp(current)

    return RegimeProbabilitySeries(
        model=fitted_model.model,
        version=fitted_model.version,
        training_observations=fitted_model.observations,
        observations=int(values.size),
        state_count=state_count,
        filtered_probabilities=tuple(tuple(float(x) for x in row) for row in filtered),
        filtered_states=tuple(int(x) for x in np.argmax(filtered, axis=1)),
        log_likelihood=float(log_likelihood),
        methodology=(
            "Causal forward filtering on a separate chronological sample with fixed "
            "parameters estimated on the prior training window. No refitting or backward "
            "smoothing is performed on evaluation observations."
        ),
    )

