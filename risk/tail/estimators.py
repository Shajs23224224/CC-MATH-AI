"""Deterministic Value-at-Risk and Expected Shortfall estimators for F22."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.stats import genpareto, norm  # type: ignore[import-untyped]

from core.contracts import TailRiskEstimate
from core.contracts.tail_risk import TailRiskMethod

_MIN_OBSERVATIONS = 20
_MIN_EVT_OBSERVATIONS = 200
_MIN_EVT_EXCEEDANCES = 20
_EPSILON = 1e-12


def _validate_returns(
    returns: Sequence[float],
    *,
    minimum: int = _MIN_OBSERVATIONS,
) -> np.ndarray:
    """Convert a one-dimensional return sample to finite float64 values."""
    try:
        values = np.asarray(tuple(float(value) for value in returns), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be a one-dimensional numeric sequence") from exc
    if values.ndim != 1:
        raise ValueError("returns must be a one-dimensional numeric sequence")
    if values.size < minimum:
        raise ValueError(f"at least {minimum} return observations are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values")
    return values


def _validate_confidence(confidence_level: float) -> float:
    """Validate a conventional one-sided tail confidence level."""
    if not math.isfinite(confidence_level) or not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and strictly between 0.5 and 1")
    return float(confidence_level)


def _validate_horizon(horizon: int) -> int:
    """Validate the number of periods in a risk horizon."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return horizon


def _estimate_from_losses(
    losses: np.ndarray,
    *,
    method: TailRiskMethod,
    confidence_level: float,
    horizon: int,
    observations: int,
    methodology: str,
    simulations: int | None = None,
    random_seed: int | None = None,
) -> TailRiskEstimate:
    """Build a common estimate record from a loss distribution."""
    if losses.ndim != 1 or losses.size == 0 or not np.all(np.isfinite(losses)):
        raise ValueError("loss scenarios must be a non-empty finite one-dimensional array")
    var = float(np.quantile(losses, confidence_level, method="higher"))
    tail = losses[losses >= var]
    if tail.size == 0:
        raise ValueError("tail distribution is empty")
    cvar = float(np.mean(tail))
    return TailRiskEstimate(
        method=method,
        var=var,
        cvar=max(cvar, var),
        confidence_level=confidence_level,
        horizon=horizon,
        observations=observations,
        simulations=simulations,
        random_seed=random_seed,
        methodology=methodology,
    )


def historical_var_cvar(
    returns: Sequence[float],
    *,
    confidence_level: float = 0.95,
    horizon: int = 1,
) -> TailRiskEstimate:
    """Estimate VaR/CVaR from rolling historical horizon returns.

    Inputs are arithmetic portfolio returns by period. Multi-period returns are
    summed as a linear P&L approximation, consistent with the parametric and
    Monte Carlo estimators; use daily horizons for the most direct interpretation.
    VaR is the empirical quantile using the conservative higher order statistic.
    """
    confidence = _validate_confidence(confidence_level)
    periods = _validate_horizon(horizon)
    values = _validate_returns(returns, minimum=_MIN_OBSERVATIONS + periods - 1)
    horizon_returns = np.asarray(
        [np.sum(values[start : start + periods]) for start in range(values.size - periods + 1)],
        dtype=np.float64,
    )
    return _estimate_from_losses(
        -horizon_returns,
        method="historical",
        confidence_level=confidence,
        horizon=periods,
        observations=int(horizon_returns.size),
        methodology=(
            "Empirical quantile of rolling horizon loss scenarios; "
            "multi-period arithmetic returns are summed; CVaR is the mean of "
            "observed losses greater than or equal to VaR."
        ),
    )


def parametric_var_cvar(
    returns: Sequence[float],
    *,
    confidence_level: float = 0.95,
    horizon: int = 1,
) -> TailRiskEstimate:
    """Estimate VaR/CVaR assuming normally distributed arithmetic returns."""
    confidence = _validate_confidence(confidence_level)
    periods = _validate_horizon(horizon)
    values = _validate_returns(returns)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= _EPSILON:
        raise ValueError("parametric VaR requires returns with positive sample variance")

    z_score = float(norm.ppf(confidence))
    horizon_mean = mean * periods
    horizon_std = standard_deviation * math.sqrt(periods)
    var = -(horizon_mean + horizon_std * z_score)
    cvar = -horizon_mean + horizon_std * float(norm.pdf(z_score)) / (1.0 - confidence)
    return TailRiskEstimate(
        method="parametric_normal",
        var=float(var),
        cvar=float(max(cvar, var)),
        confidence_level=confidence,
        horizon=periods,
        observations=int(values.size),
        methodology=(
            "Normal model fitted to arithmetic period returns; mean scales by horizon "
            "and volatility by sqrt(horizon); CVaR uses the analytical normal-tail formula."
        ),
    )


def monte_carlo_var_cvar(
    returns: Sequence[float],
    *,
    confidence_level: float = 0.95,
    horizon: int = 1,
    simulations: int = 10_000,
    random_seed: int = 0,
) -> TailRiskEstimate:
    """Estimate VaR/CVaR from reproducible Gaussian Monte Carlo return paths.

    Period returns are sampled independently from a normal distribution fitted
    to the supplied sample. Horizon P&L is additive, matching the parametric
    estimator's arithmetic-return approximation.
    """
    confidence = _validate_confidence(confidence_level)
    periods = _validate_horizon(horizon)
    values = _validate_returns(returns)
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 100:
        raise ValueError("simulations must be an integer greater than or equal to 100")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= _EPSILON:
        raise ValueError("Monte Carlo VaR requires returns with positive sample variance")

    generator = np.random.default_rng(random_seed)
    simulated_period_returns = generator.normal(
        loc=mean,
        scale=standard_deviation,
        size=(simulations, periods),
    )
    losses = -np.sum(simulated_period_returns, axis=1)
    return _estimate_from_losses(
        losses,
        method="monte_carlo_normal",
        confidence_level=confidence,
        horizon=periods,
        observations=int(values.size),
        simulations=simulations,
        random_seed=random_seed,
        methodology=(
            "Independent Gaussian period-return draws fitted to the sample; horizon P&L is "
            "the sum of simulated arithmetic returns; NumPy Generator with explicit seed."
        ),
    )


def evt_var_cvar(
    returns: Sequence[float],
    *,
    confidence_level: float = 0.99,
    threshold_quantile: float = 0.90,
) -> TailRiskEstimate:
    """Estimate extreme-tail VaR/CVaR with a Peaks-Over-Threshold GPD fit.

    EVT is intentionally conservative about data requirements: it needs at least
    200 observations, 20 exceedances over the selected loss threshold, a target
    confidence above the threshold quantile, and a fitted shape parameter below
    one so the expected shortfall is finite.
    """
    confidence = _validate_confidence(confidence_level)
    if not math.isfinite(threshold_quantile) or not 0.5 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be finite and strictly between 0.5 and 1")
    if confidence <= threshold_quantile:
        raise ValueError("confidence_level must exceed threshold_quantile for EVT extrapolation")

    values = _validate_returns(returns, minimum=_MIN_EVT_OBSERVATIONS)
    losses = -values
    threshold = float(np.quantile(losses, threshold_quantile, method="higher"))
    exceedances = losses[losses > threshold] - threshold
    exceedance_count = int(exceedances.size)
    if exceedance_count < _MIN_EVT_EXCEEDANCES:
        raise ValueError(
            f"EVT requires at least {_MIN_EVT_EXCEEDANCES} threshold exceedances"
        )

    shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    shape = float(shape)
    scale = float(scale)
    if not math.isfinite(shape) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("EVT fit returned invalid parameters")
    if shape >= 1.0:
        raise ValueError("EVT expected shortfall is not finite when GPD shape is >= 1")

    tail_probability = 1.0 - confidence
    empirical_exceedance_probability = exceedance_count / values.size
    ratio = empirical_exceedance_probability / tail_probability
    if ratio <= 1.0:
        raise ValueError("insufficient exceedances to extrapolate to the requested confidence")

    if abs(shape) < 1e-8:
        var = threshold + scale * math.log(ratio)
    else:
        var = threshold + (scale / shape) * (ratio**shape - 1.0)
    cvar = (var + scale - shape * threshold) / (1.0 - shape)
    if not math.isfinite(var) or not math.isfinite(cvar) or cvar < var:
        raise ValueError("EVT produced an invalid VaR/CVaR estimate")

    return TailRiskEstimate(
        method="evt_peaks_over_threshold",
        var=float(var),
        cvar=float(cvar),
        confidence_level=confidence,
        horizon=1,
        observations=int(values.size),
        methodology=(
            "Peaks-Over-Threshold generalized Pareto fit to losses above the empirical "
            f"{threshold_quantile:.3f} quantile; {exceedance_count} exceedances; "
            f"GPD shape={shape:.6g}, scale={scale:.6g}."
        ),
    )
