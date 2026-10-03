import math

import pytest

from core.contracts import (
    DynamicCorrelationForecast,
    MeanReversionFit,
    MeanReversionForecast,
    VolatilityFitSummary,
    VolatilityForecast,
)
from quant.stochastic import fit_mean_reversion, fit_ornstein_uhlenbeck
from quant.volatility import (
    ewma_volatility,
    fit_arch,
    fit_dcc_garch,
    fit_egarch,
    fit_garch,
    fit_gjr_garch,
    fit_stochastic_volatility,
    rolling_volatility,
)


def _returns(length: int = 180) -> tuple[float, ...]:
    return tuple(
        0.001 * math.sin(index / 5.0)
        + 0.007 * math.sin(index / 2.7)
        + 0.003 * math.cos(index / 11.0)
        for index in range(length)
    )


def _mean_reverting(length: int = 160) -> tuple[float, ...]:
    values = [0.0]
    for index in range(1, length):
        innovation = 0.15 * math.sin(index / 3.0) + 0.03 * math.cos(index / 5.0)
        values.append(0.82 * values[-1] + innovation)
    return tuple(values)


def _multivariate_returns(length: int = 120) -> tuple[tuple[float, float], ...]:
    base = _returns(length)
    return tuple(
        (value, 0.65 * value + 0.002 * math.cos(index / 7.0)) for index, value in enumerate(base)
    )


def test_rolling_and_ewma_volatility() -> None:
    sample = (0.01, 0.02, 0.03, 0.04)
    expected = 0.005 * math.sqrt(252.0)
    assert rolling_volatility(sample, window=2, annualization_factor=252.0) == pytest.approx(
        (expected, expected, expected)
    )
    values = _returns()
    rolling = rolling_volatility(values, window=20)
    ewma = ewma_volatility(values, decay=0.94)
    assert len(rolling) == len(values) - 20 + 1
    assert len(ewma) == len(values)
    assert all(value >= 0.0 and math.isfinite(value) for value in rolling)
    assert all(value >= 0.0 and math.isfinite(value) for value in ewma)
    assert rolling_volatility(values, 20, 252.0) == rolling_volatility(values, 20, 252.0)


def test_conditional_volatility_models_return_finite_forecasts() -> None:
    values = _returns()
    fitted_models = (
        fit_arch(values),
        fit_garch(values),
        fit_egarch(values),
        fit_gjr_garch(values),
        fit_stochastic_volatility(values),
    )
    for fitted in fitted_models:
        summary = fitted.summary()
        forecast = fitted.forecast(4)
        assert isinstance(summary, VolatilityFitSummary)
        assert isinstance(forecast, VolatilityForecast)
        assert summary.converged
        assert len(forecast.values) == 4
        assert all(math.isfinite(value) and value >= 0.0 for value in forecast.values)


def test_dcc_garch_preserves_dynamic_correlation_contract() -> None:
    fitted = fit_dcc_garch(_multivariate_returns())
    forecast = fitted.correlation_forecast()
    assert isinstance(forecast, DynamicCorrelationForecast)
    assert forecast.horizon == 1
    assert len(forecast.matrix) == 2
    assert forecast.matrix[0][0] == pytest.approx(1.0)
    assert forecast.matrix[1][1] == pytest.approx(1.0)
    assert forecast.matrix[0][1] == pytest.approx(forecast.matrix[1][0])
    assert -1.0 <= forecast.matrix[0][1] <= 1.0


def test_ornstein_uhlenbeck_and_mean_reversion() -> None:
    values = _mean_reverting()
    for fitted in (fit_ornstein_uhlenbeck(values), fit_mean_reversion(values)):
        summary = fitted.summary()
        forecast = fitted.forecast(5)
        assert isinstance(summary, MeanReversionFit)
        assert isinstance(forecast, MeanReversionForecast)
        assert summary.speed > 0.0
        assert summary.half_life > 0.0
        assert len(forecast.values) == 5
        assert all(math.isfinite(value) for value in forecast.values)


def test_volatility_boundaries_reject_invalid_configuration() -> None:
    values = _returns()
    with pytest.raises(ValueError, match="window"):
        rolling_volatility(values, 1)
    with pytest.raises(ValueError, match="decay"):
        ewma_volatility(values, decay=1.0)
    with pytest.raises(ValueError, match="annualization_factor"):
        fit_garch(values, annualization_factor=0.0)
    with pytest.raises(ValueError, match="horizon"):
        fit_garch(values).forecast(0)
    with pytest.raises(ValueError, match="positive"):
        fit_ornstein_uhlenbeck(values, dt=0.0)


def test_contracts_are_strict_and_finite() -> None:
    with pytest.raises(ValueError):
        VolatilityForecast(
            model="garch",
            horizon=2,
            annualization_factor=252.0,
            values=(0.1,),
        )
    with pytest.raises(ValueError):
        DynamicCorrelationForecast(
            model="dcc_garch",
            horizon=1,
            matrix=((1.0, 1.2), (1.2, 1.0)),
        )
    with pytest.raises(ValueError):
        MeanReversionForecast(
            model="mean_reversion",
            horizon=2,
            values=(0.1, math.inf),
        )
