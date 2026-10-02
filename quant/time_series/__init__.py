"""Deterministic time-series modelling API."""

from .diagnostics import ljung_box_test, residual_diagnostics
from .models import (
    FittedTimeSeriesModel,
    fit_ar,
    fit_arima,
    fit_arimax,
    fit_arma,
    fit_exponential_smoothing,
    fit_holt,
    fit_holt_winters,
    fit_ma,
    fit_sarima,
    fit_var,
    fit_vecm,
)
from .validation import (
    temporal_train_test_split,
    validate_exogenous,
    validate_forecast_horizon,
    validate_lag,
    validate_matrix,
    validate_order,
    validate_seasonal_order,
    validate_univariate,
)

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
    "ljung_box_test",
    "residual_diagnostics",
    "temporal_train_test_split",
    "validate_exogenous",
    "validate_forecast_horizon",
    "validate_lag",
    "validate_matrix",
    "validate_order",
    "validate_seasonal_order",
    "validate_univariate",
]
