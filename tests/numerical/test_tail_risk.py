"""Numerical reference tests for F22 Tail Risk."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from risk.tail import (
    evt_var_cvar,
    historical_var_cvar,
    monte_carlo_var_cvar,
    parametric_var_cvar,
)


def test_historical_var_and_cvar_reference_values() -> None:
    returns = (0.01,) * 17 + (-0.02, -0.03, -0.04)

    estimate = historical_var_cvar(returns, confidence_level=0.90)

    assert estimate.var == pytest.approx(0.03)
    assert estimate.cvar == pytest.approx(0.035)
    assert estimate.method == "historical"
    assert estimate.observations == 20
    assert estimate.unit == "fraction_of_notional"


def test_historical_horizon_uses_rolling_arithmetic_return_sums() -> None:
    returns = tuple(float(value) for value in range(-10, 11))
    estimate = historical_var_cvar(returns, confidence_level=0.90, horizon=2)

    assert estimate.horizon == 2
    assert estimate.observations == 20
    assert estimate.var <= estimate.cvar


def test_parametric_var_cvar_matches_normal_tail_formula() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(-0.001, 0.02, size=500)
    confidence = 0.95
    horizon = 3
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    z_score = float(norm.ppf(confidence))

    estimate = parametric_var_cvar(
        tuple(returns),
        confidence_level=confidence,
        horizon=horizon,
    )

    assert estimate.var == pytest.approx(-mean * horizon + std * np.sqrt(horizon) * z_score)
    assert estimate.cvar == pytest.approx(
        -mean * horizon + std * np.sqrt(horizon) * float(norm.pdf(z_score)) / (1.0 - confidence)
    )


def test_monte_carlo_is_reproducible_and_records_seed() -> None:
    returns = tuple(np.random.default_rng(44).normal(0.0002, 0.012, size=250))
    first = monte_carlo_var_cvar(
        returns,
        confidence_level=0.99,
        horizon=3,
        simulations=5_000,
        random_seed=123,
    )
    second = monte_carlo_var_cvar(
        returns,
        confidence_level=0.99,
        horizon=3,
        simulations=5_000,
        random_seed=123,
    )

    assert first == second
    assert first.method == "monte_carlo_normal"
    assert first.simulations == 5_000
    assert first.random_seed == 123
    assert first.cvar >= first.var


def test_evt_peaks_over_threshold_estimates_extreme_tail() -> None:
    rng = np.random.default_rng(77)
    returns = tuple((rng.standard_t(df=4, size=3_000) * 0.01).tolist())

    estimate = evt_var_cvar(
        returns,
        confidence_level=0.99,
        threshold_quantile=0.90,
    )

    assert estimate.method == "evt_peaks_over_threshold"
    assert estimate.observations == 3_000
    assert estimate.horizon == 1
    assert estimate.cvar >= estimate.var


@pytest.mark.parametrize(
    ("returns", "kwargs", "message"),
    [
        ((0.01,) * 19, {}, "at least 20"),
        ((0.01,) * 19 + (float("nan"),), {}, "finite"),
        ((0.01,) * 20, {}, "positive sample variance"),
        ((0.01,) * 20, {"confidence_level": 1.0}, "confidence_level"),
        ((0.01,) * 20, {"horizon": 0}, "horizon"),
        ((0.01,) * 20, {"simulations": 50}, "simulations"),
        ((0.01,) * 20, {"random_seed": -1}, "random_seed"),
    ],
)
def test_invalid_inputs_are_rejected(
    returns: tuple[float, ...],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        if "simulations" in kwargs or "random_seed" in kwargs:
            monte_carlo_var_cvar(returns, **kwargs)  # type: ignore[arg-type]
        else:
            parametric_var_cvar(returns, **kwargs)  # type: ignore[arg-type]


def test_evt_rejects_insufficient_data_and_threshold_mismatch() -> None:
    with pytest.raises(ValueError, match="at least 200"):
        evt_var_cvar((0.01,) * 100)
    returns = tuple(np.random.default_rng(11).normal(0.0, 0.01, size=500))
    with pytest.raises(ValueError, match="must exceed threshold_quantile"):
        evt_var_cvar(returns, confidence_level=0.90, threshold_quantile=0.90)
