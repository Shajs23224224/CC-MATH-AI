import math

import pytest

from core.contracts import ResidualDiagnostics, TemporalSplit, TimeSeriesForecast
from quant.time_series import (
    fit_ar,
    fit_arimax,
    fit_arma,
    fit_exponential_smoothing,
    fit_holt,
    fit_holt_winters,
    fit_ma,
    fit_sarima,
    fit_var,
    fit_vecm,
    residual_diagnostics,
    temporal_train_test_split,
)


def _series(length: int = 32) -> tuple[float, ...]:
    return tuple(100.0 + 0.6 * index + 2.0 * math.sin(index / 3.0) for index in range(length))


def _multivariate(length: int = 40) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            100.0 + 0.3 * index + math.sin(index / 4.0),
            80.0 + 0.3 * index + math.sin(index / 4.0) + 0.4 * math.sin(index / 2.5),
        )
        for index in range(length)
    )


def test_temporal_split_is_chronological() -> None:
    train, test, split = temporal_train_test_split(_series(20), 14)
    assert train[-1] == 100.0 + 0.6 * 13 + 2.0 * math.sin(13 / 3.0)
    assert test[0] == 100.0 + 0.6 * 14 + 2.0 * math.sin(14 / 3.0)
    assert isinstance(split, TemporalSplit)
    assert split.train_size == 14
    assert split.test_size == 6


def test_ar_ma_arma_produce_typed_forecasts() -> None:
    values = _series()
    for fitted in (
        fit_ar(values),
        fit_ma(values),
        fit_arma(values, 1, 1),
        fit_arima(values, (1, 1, 0)),
    ):
        forecast = fitted.forecast(4)
        assert isinstance(forecast, TimeSeriesForecast)
        assert forecast.horizon == 4
        assert len(forecast.values) == 4
        assert all(math.isfinite(value) for value in forecast.values)


def test_sarima_and_arimax_require_future_exogenous_information() -> None:
    values = _series()
    sarima = fit_sarima(values, order=(1, 0, 0), seasonal_order=(0, 1, 1, 4))
    assert sarima.forecast(3).horizon == 3

    exog = tuple((0.1 + index / 100.0,) for index in range(len(values)))
    arimax = fit_arimax(values, exog, order=(1, 0, 0))
    with pytest.raises(ValueError, match="future_exog"):
        arimax.forecast(2)
    future = arimax.forecast(2, future_exog=((0.5,), (0.6,)))
    assert len(future.values) == 2


def test_exponential_smoothing_variants_are_deterministic() -> None:
    values = _series()
    fits = [
        fit_exponential_smoothing(values),
        fit_holt(values),
        fit_holt_winters(values, seasonal_periods=4),
    ]
    forecasts = [fitted.forecast(3).values for fitted in fits]
    for forecast in forecasts:
        assert len(forecast) == 3
        assert all(math.isfinite(value) for value in forecast)
    assert forecasts[0] == fit_exponential_smoothing(values).forecast(3).values
    assert forecasts[1] == fit_holt(values).forecast(3).values


def test_var_and_vecm_retain_multivariate_forecasts() -> None:
    values = _multivariate()
    var = fit_var(values, lags=1)
    vecm = fit_vecm(values, k_ar_diff=1, coint_rank=1)
    var_forecast = var.forecast(2)
    vecm_forecast = vecm.forecast(2)
    assert var_forecast.horizon == 2
    assert vecm_forecast.horizon == 2
    assert var_forecast.multivariate_values is not None
    assert vecm_forecast.multivariate_values is not None
    assert len(var_forecast.multivariate_values) == 2
    assert len(vecm_forecast.multivariate_values[0]) == 2
    assert len(vecm_forecast.values) == 2


def test_residual_diagnostics_and_fit_summary() -> None:
    fitted = fit_ar(_series())
    summary = fitted.summary()
    diagnostics = fitted.residual_diagnostics(lag=6)
    assert summary.model == "ar"
    assert summary.observations == 32
    assert summary.parameter_count >= 1
    assert isinstance(diagnostics, ResidualDiagnostics)
    assert 0.0 <= diagnostics.ljung_box_pvalue <= 1.0
    multivariate = fit_var(_multivariate())
    assert 0.0 <= multivariate.residual_diagnostics(series_index=1).ljung_box_pvalue <= 1.0


def test_validation_boundaries() -> None:
    with pytest.raises(ValueError, match="minimum"):
        fit_ar((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="positive integer"):
        fit_ar(_series(), lags=0)
    with pytest.raises(ValueError, match="seasonal period"):
        fit_sarima(_series(), seasonal_order=(0, 1, 1, 1))
    with pytest.raises(ValueError, match="coint_rank"):
        fit_vecm(_multivariate(), coint_rank=2)


def test_stable_residual_diagnostic_pvalue() -> None:
    residuals = tuple(math.sin(index / 2.0) for index in range(40))
    diagnostics = residual_diagnostics(residuals, lag=5)
    assert diagnostics.ljung_box_statistic >= 0.0
    assert 0.0 <= diagnostics.ljung_box_pvalue <= 1.0
