"""Deterministic time-series model adapters built on statsmodels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg  # type: ignore[import-untyped]
from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-untyped]
from statsmodels.tsa.api import VAR  # type: ignore[import-untyped]
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt  # type: ignore[import-untyped]
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]
from statsmodels.tsa.vector_ar.vecm import VECM  # type: ignore[import-untyped]

from core.contracts import TimeSeriesFamily, TimeSeriesForecast, TimeSeriesFitSummary

from .validation import (
    validate_exogenous,
    validate_forecast_horizon,
    validate_lag,
    validate_matrix,
    validate_order,
    validate_seasonal_order,
    validate_univariate,
)


@dataclass(frozen=True)
class FittedTimeSeriesModel:
    """Typed facade over a fitted statsmodels result."""

    model: TimeSeriesFamily
    result: Any
    observations: int

    def summary(self) -> TimeSeriesFitSummary:
        """Return stable fit metadata without exposing estimator internals."""
        params = getattr(self.result, "params", ())
        parameter_count = int(np.asarray(params).size)
        converged = getattr(self.result, "converged", None)
        if converged is None:
            converged = bool(getattr(self.result, "mle_retvals", {}).get("converged", True))
        aic = _finite_optional(getattr(self.result, "aic", None))
        bic = _finite_optional(getattr(self.result, "bic", None))
        return TimeSeriesFitSummary(
            model=self.model,
            observations=self.observations,
            parameter_count=parameter_count,
            aic=aic,
            bic=bic,
            converged=bool(converged),
        )

    def forecast(
        self,
        horizon: int,
        future_exog: Sequence[Sequence[float]] | None = None,
    ) -> TimeSeriesForecast:
        """Produce a typed out-of-sample forecast."""
        steps = validate_forecast_horizon(horizon)
        if self.model in {"var", "vecm"}:
            values = self._forecast_multivariate(steps)
            return TimeSeriesForecast(model=self.model, horizon=steps, values=values)

        kwargs: dict[str, Any] = {}
        if self.model == "arimax":
            if future_exog is None:
                raise ValueError("future_exog is required for ARIMAX forecasts")
            checked = validate_exogenous(future_exog, steps)
            kwargs["exog"] = np.asarray(checked, dtype=float)
        values = np.asarray(self.result.forecast(steps=steps, **kwargs), dtype=float).reshape(-1)
        flattened = tuple(float(value) for value in values)
        return TimeSeriesForecast(model=self.model, horizon=steps, values=flattened)

    def residual_diagnostics(self, lag: int = 10):
        """Return deterministic residual diagnostics."""
        from .diagnostics import residual_diagnostics

        residuals = np.asarray(self.result.resid, dtype=float).reshape(-1)
        return residual_diagnostics(tuple(float(value) for value in residuals), lag=lag)

    def _forecast_multivariate(self, horizon: int) -> tuple[float, ...]:
        if self.model == "var":
            endog = np.asarray(self.result.endog, dtype=float)
            values = np.asarray(
                self.result.forecast(endog[-self.result.k_ar :], steps=horizon),
                dtype=float,
            )
        else:
            values = np.asarray(self.result.predict(steps=horizon), dtype=float)
        if values.ndim != 2 or values.shape[0] != horizon:
            raise RuntimeError("multivariate forecast has an unexpected shape")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("multivariate forecast contains non-finite values")
        return tuple(float(value) for value in values[:, 0])


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _fit_arima(
    values: tuple[float, ...],
    order: tuple[int, int, int],
) -> Any:
    model = ARIMA(np.asarray(values, dtype=float), order=order, trend="n")
    return model.fit()


def fit_ar(values: Sequence[float], lags: int = 1) -> FittedTimeSeriesModel:
    checked = validate_univariate(values, minimum=max(8, lags + 3))
    validate_lag(lags)
    result = _fit_arima(checked, (lags, 0, 0))
    return FittedTimeSeriesModel("ar", result, len(checked))


def fit_ma(values: Sequence[float], order: int = 1) -> FittedTimeSeriesModel:
    checked = validate_univariate(values, minimum=max(8, order + 3))
    validate_lag(order)
    result = _fit_arima(checked, (0, 0, order))
    return FittedTimeSeriesModel("ma", result, len(checked))


def fit_arma(values: Sequence[float], p: int = 1, q: int = 1) -> FittedTimeSeriesModel:
    checked = validate_univariate(values, minimum=max(10, p + q + 4))
    validate_lag(p)
    validate_lag(q)
    result = _fit_arima(checked, (p, 0, q))
    return FittedTimeSeriesModel("arma", result, len(checked))


def fit_arima(
    values: Sequence[float],
    order: tuple[int, int, int] = (1, 1, 0),
) -> FittedTimeSeriesModel:
    checked = validate_univariate(values, minimum=max(10, sum(order) + 5))
    checked_order = validate_order(order)
    result = _fit_arima(checked, checked_order)
    return FittedTimeSeriesModel("arima", result, len(checked))


def fit_sarima(
    values: Sequence[float],
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 4),
) -> FittedTimeSeriesModel:
    checked = validate_univariate(
        values,
        minimum=max(16, sum(order) + sum(seasonal_order[:3]) + 5),
    )
    checked_order = validate_order(order)
    checked_seasonal = validate_seasonal_order(seasonal_order)
    model = SARIMAX(
        np.asarray(checked, dtype=float),
        order=checked_order,
        seasonal_order=checked_seasonal,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return FittedTimeSeriesModel("sarima", result, len(checked))


def fit_arimax(
    values: Sequence[float],
    exog: Sequence[Sequence[float]],
    order: tuple[int, int, int] = (1, 0, 0),
) -> FittedTimeSeriesModel:
    checked = validate_univariate(values, minimum=max(10, sum(order) + 5))
    checked_exog = validate_exogenous(exog, len(checked))
    checked_order = validate_order(order)
    model = SARIMAX(
        np.asarray(checked, dtype=float),
        exog=np.asarray(checked_exog, dtype=float),
        order=checked_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return FittedTimeSeriesModel("arimax", result, len(checked))


def fit_exponential_smoothing(
    values: Sequence[float],
) -> FittedTimeSeriesModel:
    checked = validate_univariate(values, minimum=4)
    result = ExponentialSmoothing(np.asarray(checked, dtype=float)).fit()
    return FittedTimeSeriesModel("exponential_smoothing", result, len(checked))


def fit_holt(
    values: Sequence[float],
    damped_trend: bool = False,
) -> FittedTimeSeriesModel:
    checked = validate_univariate(values, minimum=5)
    result = Holt(np.asarray(checked, dtype=float), damped_trend=damped_trend).fit()
    return FittedTimeSeriesModel("holt", result, len(checked))


def fit_holt_winters(
    values: Sequence[float],
    seasonal_periods: int,
    trend: str | None = "add",
    seasonal: str | None = "add",
) -> FittedTimeSeriesModel:
    checked = validate_univariate(
        values,
        minimum=max(2 * validate_lag(seasonal_periods), 8),
    )
    result = ExponentialSmoothing(
        np.asarray(checked, dtype=float),
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    ).fit()
    return FittedTimeSeriesModel("holt_winters", result, len(checked))


def fit_var(
    values: Sequence[Sequence[float]],
    lags: int = 1,
) -> FittedTimeSeriesModel:
    checked = validate_matrix(values)
    validate_lag(lags)
    if len(checked) <= lags + 1:
        raise ValueError("VAR lag order leaves insufficient observations")
    model = VAR(np.asarray(checked, dtype=float))
    result = model.fit(lags)
    return FittedTimeSeriesModel("var", result, len(checked))


def fit_vecm(
    values: Sequence[Sequence[float]],
    k_ar_diff: int = 1,
    coint_rank: int = 1,
) -> FittedTimeSeriesModel:
    checked = validate_matrix(values)
    validate_lag(k_ar_diff)
    if not 1 <= coint_rank < len(checked[0]):
        raise ValueError("coint_rank must be between 1 and number of series minus 1")
    if len(checked) <= k_ar_diff + 2:
        raise ValueError("VECM lag configuration leaves insufficient observations")
    model = VECM(
        np.asarray(checked, dtype=float),
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic="n",
    )
    result = model.fit()
    return FittedTimeSeriesModel("vecm", result, len(checked))


__all__ = [
    "FittedTimeSeriesModel",
    "fit_ar",
    "fit_arima",
    "fit_arimax",
    "fit_arma",
    "fit_exponential_smoothing",
    "fit_holt",
    "fit_holt_winters",
    "fit_ma",
    "fit_sarima",
    "fit_var",
    "fit_vecm",
]
