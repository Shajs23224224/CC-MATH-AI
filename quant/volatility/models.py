"""Deterministic volatility models built on NumPy and SciPy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from core.contracts import (
    DynamicCorrelationForecast,
    VolatilityFamily,
    VolatilityFitSummary,
    VolatilityForecast,
)

_EPSILON = 1e-12


def _validate_returns(values: Sequence[float], minimum: int = 30) -> np.ndarray:
    checked = np.asarray(tuple(float(value) for value in values), dtype=float)
    if checked.size < minimum:
        raise ValueError(f"at least {minimum} return observations are required")
    if not np.all(np.isfinite(checked)):
        raise ValueError("returns must be finite")
    if float(np.var(checked)) <= _EPSILON:
        raise ValueError("returns must have positive variance")
    return checked - float(np.mean(checked))


def _validate_finite_series(values: Sequence[float], minimum: int) -> np.ndarray:
    checked = np.asarray(tuple(float(value) for value in values), dtype=float)
    if checked.size < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    if not np.all(np.isfinite(checked)):
        raise ValueError("observations must be finite")
    return checked


def _validate_annualization_factor(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("annualization_factor must be finite and positive")
    return float(value)


def rolling_volatility(
    returns: Sequence[float],
    window: int,
    annualization_factor: float = 252.0,
) -> tuple[float, ...]:
    """Return trailing population volatility annualized by sqrt(frequency)."""
    if window < 2:
        raise ValueError("window must be at least 2")
    annualization = _validate_annualization_factor(annualization_factor)
    checked = _validate_finite_series(returns, minimum=window)
    result: list[float] = []
    for index in range(window, len(checked) + 1):
        chunk = checked[index - window : index]
        result.append(float(np.sqrt(np.var(chunk) * annualization)))
    return tuple(result)


def ewma_volatility(
    returns: Sequence[float],
    decay: float = 0.94,
    annualization_factor: float = 252.0,
) -> tuple[float, ...]:
    """Return recursively weighted volatility using the RiskMetrics-style decay."""
    if not math.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("decay must be strictly between 0 and 1")
    annualization = _validate_annualization_factor(annualization_factor)
    checked = _validate_finite_series(returns, minimum=2)
    variance = float(checked[0] ** 2)
    result = [math.sqrt(max(variance, 0.0) * annualization)]
    for value in checked[1:]:
        variance = decay * variance + (1.0 - decay) * float(value**2)
        result.append(math.sqrt(max(variance, 0.0) * annualization))
    return tuple(result)


def _normal_log_likelihood(errors: np.ndarray, variances: np.ndarray) -> float:
    safe_variance = np.maximum(variances, _EPSILON)
    log_terms = np.log(2.0 * math.pi) + np.log(safe_variance) + errors * errors / safe_variance
    return float(-0.5 * np.sum(log_terms))


@dataclass(frozen=True)
class FittedVolatilityModel:
    """Typed facade over a fitted conditional-volatility model."""

    model: VolatilityFamily
    parameters: tuple[float, ...]
    log_likelihood: float
    observations: int
    converged: bool
    annualization_factor: float
    forecast_kind: str
    last_variance: float
    last_error: float

    def summary(self) -> VolatilityFitSummary:
        """Return stable fit metadata."""
        parameter_count = len(self.parameters)
        aic = -2.0 * self.log_likelihood + 2.0 * parameter_count
        bic = -2.0 * self.log_likelihood + math.log(self.observations) * parameter_count
        return VolatilityFitSummary(
            model=self.model,
            observations=self.observations,
            parameter_count=parameter_count,
            log_likelihood=self.log_likelihood,
            aic=aic,
            bic=bic,
            converged=self.converged,
        )

    def forecast(self, horizon: int) -> VolatilityForecast:
        """Return deterministic conditional-volatility forecasts."""
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if self.forecast_kind in {"arch", "garch", "gjr_garch"}:
            omega, alpha = self.parameters[0], self.parameters[1]
            beta = self.parameters[2] if self.forecast_kind != "arch" else 0.0
            gamma = self.parameters[3] if self.forecast_kind == "gjr_garch" else 0.0
            variance = self.last_variance
            error = self.last_error
            values: list[float] = []
            for _ in range(horizon):
                leverage = gamma * error**2 if error < 0.0 else 0.0
                variance = max(omega + alpha * error**2 + beta * variance + leverage, _EPSILON)
                values.append(math.sqrt(variance * self.annualization_factor))
                error = 0.0
            return VolatilityForecast(
                model=self.model,
                horizon=horizon,
                annualization_factor=self.annualization_factor,
                values=tuple(values),
            )
        if self.forecast_kind == "egarch":
            omega, alpha, gamma, beta = self.parameters
            log_variance = math.log(max(self.last_variance, _EPSILON))
            expectation_abs_z = math.sqrt(2.0 / math.pi)
            values = []
            for _ in range(horizon):
                log_variance = omega + beta * log_variance - alpha * expectation_abs_z
                values.append(math.sqrt(math.exp(log_variance) * self.annualization_factor))
            return VolatilityForecast(
                model=self.model,
                horizon=horizon,
                annualization_factor=self.annualization_factor,
                values=tuple(values),
            )
        if self.forecast_kind == "stochastic_volatility":
            mu, phi, _eta_std = self.parameters
            initial_log_variance = math.log(max(self.last_variance, _EPSILON))
            forecast_values = tuple(
                math.sqrt(
                    math.exp(mu + (initial_log_variance - mu) * math.pow(phi, step))
                    * self.annualization_factor
                )
                for step in range(1, horizon + 1)
            )
            return VolatilityForecast(
                model=self.model,
                horizon=horizon,
                annualization_factor=self.annualization_factor,
                values=forecast_values,
            )
        raise ValueError("unsupported univariate forecast model")


@dataclass(frozen=True)
class FittedDCCGARCHModel:
    """Typed DCC-GARCH fit preserving standardized shocks and correlation state."""

    models: tuple[FittedVolatilityModel, ...]
    dcc_alpha: float
    dcc_beta: float
    q_bar: tuple[tuple[float, ...], ...]
    q_last: tuple[tuple[float, ...], ...]
    z_last: tuple[float, ...]

    def correlation_forecast(self, horizon: int = 1) -> DynamicCorrelationForecast:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if horizon != 1:
            raise ValueError("DCC-GARCH currently supports one-step correlation forecasting")
        q_bar = np.asarray(self.q_bar, dtype=float)
        q_last = np.asarray(self.q_last, dtype=float)
        z_last = np.asarray(self.z_last, dtype=float)
        q_next = (
            (1.0 - self.dcc_alpha - self.dcc_beta) * q_bar
            + self.dcc_alpha * np.outer(z_last, z_last)
            + self.dcc_beta * q_last
        )
        diagonal = np.sqrt(np.maximum(np.diag(q_next), _EPSILON))
        correlation = q_next / np.outer(diagonal, diagonal)
        correlation = 0.5 * (correlation + correlation.T)
        np.fill_diagonal(correlation, 1.0)
        return DynamicCorrelationForecast(
            model="dcc_garch",
            horizon=1,
            matrix=tuple(tuple(float(value) for value in row) for row in correlation),
        )

    def univariate_forecasts(self, horizon: int = 1) -> tuple[VolatilityForecast, ...]:
        return tuple(model.forecast(horizon) for model in self.models)


def _optimize(
    objective: Any,
    initial: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    constraints: Sequence[dict[str, Any]] = (),
    method: str = "SLSQP",
) -> tuple[tuple[float, ...], float, bool]:
    optimize_kwargs: dict[str, Any] = {
        "method": method,
        "bounds": list(bounds),
        "options": {"maxiter": 1000, "ftol": 1e-10},
    }
    if method == "SLSQP":
        optimize_kwargs["constraints"] = list(constraints)
    result = minimize(
        objective,
        np.asarray(tuple(initial), dtype=float),
        **optimize_kwargs,
    )
    if not bool(result.success):
        raise RuntimeError(f"volatility optimization failed to converge: {result.message}")
    parameters = tuple(float(value) for value in result.x)
    return parameters, float(-result.fun), bool(result.success)


def _conditional_variances(
    errors: np.ndarray,
    parameters: tuple[float, ...],
) -> np.ndarray:
    variance = float(np.var(errors))
    values = np.empty(errors.size, dtype=float)
    values[0] = variance
    for index in range(1, errors.size):
        values[index] = (
            parameters[0]
            + parameters[1] * errors[index - 1] ** 2
            + parameters[2] * values[index - 1]
        )
    return np.maximum(values, _EPSILON)


def _garch_fit(
    returns: Sequence[float],
    kind: str,
    annualization_factor: float,
) -> FittedVolatilityModel:
    errors = _validate_returns(returns, minimum=40)
    annualization = _validate_annualization_factor(annualization_factor)
    variance = float(np.var(errors))
    constraints: tuple[dict[str, Any], ...] = ()
    initial: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]

    if kind == "arch":
        initial = (0.1 * variance, 0.85)
        bounds = ((_EPSILON, 10.0 * variance), (1e-8, 0.999))

        def objective(params: np.ndarray) -> float:
            omega, alpha = params
            values = np.empty(errors.size, dtype=float)
            values[0] = variance
            for index in range(1, errors.size):
                values[index] = omega + alpha * errors[index - 1] ** 2
            return -_normal_log_likelihood(errors, values)
    elif kind == "garch":
        initial = (0.05 * variance, 0.05, 0.90)
        bounds = ((_EPSILON, 10.0 * variance), (1e-8, 0.999), (0.0, 0.999))
        constraints = ({"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]},)

        def objective(params: np.ndarray) -> float:
            omega, alpha, beta = params
            values = np.empty(errors.size, dtype=float)
            values[0] = variance
            for index in range(1, errors.size):
                values[index] = omega + alpha * errors[index - 1] ** 2 + beta * values[index - 1]
            return -_normal_log_likelihood(errors, values)
    elif kind == "egarch":
        initial = (math.log(max(0.05 * variance, _EPSILON)), 0.10, 0.0, 0.90)
        bounds = ((-10.0, 10.0), (-5.0, 5.0), (-5.0, 5.0), (-0.999, 0.999))

        def objective(params: np.ndarray) -> float:
            omega, alpha, gamma, beta = params
            log_values = np.empty(errors.size, dtype=float)
            log_values[0] = math.log(max(variance, _EPSILON))
            expectation_abs_z = math.sqrt(2.0 / math.pi)
            for index in range(1, errors.size):
                previous_log_variance = log_values[index - 1]
                previous_std = math.exp(0.5 * np.clip(previous_log_variance, -50.0, 50.0))
                z = errors[index - 1] / previous_std
                log_values[index] = (
                    omega
                    + beta * previous_log_variance
                    + alpha * (abs(z) - expectation_abs_z)
                    + gamma * z
                )
                if not math.isfinite(float(log_values[index])):
                    return 1e12
            safe_log_variance = np.clip(log_values, -50.0, 50.0)
            log_terms = (
                math.log(2.0 * math.pi)
                + safe_log_variance
                + errors * errors * np.exp(-safe_log_variance)
            )
            if not np.all(np.isfinite(log_terms)):
                return 1e12
            return float(0.5 * np.sum(log_terms))
    elif kind == "gjr_garch":
        initial = (0.05 * variance, 0.04, 0.05, 0.90)
        bounds = ((_EPSILON, 10.0 * variance), (0.0, 0.999), (0.0, 0.999), (0.0, 0.999))
        constraints = ({"type": "ineq", "fun": lambda p: 0.999 - p[1] - 0.5 * p[2] - p[3]},)

        def objective(params: np.ndarray) -> float:
            omega, alpha, gamma, beta = params
            values = np.empty(errors.size, dtype=float)
            values[0] = variance
            for index in range(1, errors.size):
                indicator = 1.0 if errors[index - 1] < 0.0 else 0.0
                values[index] = (
                    omega
                    + alpha * errors[index - 1] ** 2
                    + gamma * indicator * errors[index - 1] ** 2
                    + beta * values[index - 1]
                )
            return -_normal_log_likelihood(errors, values)
    else:
        raise ValueError(f"unsupported volatility model: {kind}")

    optimization_method = "L-BFGS-B" if kind == "egarch" else "SLSQP"
    parameters, log_likelihood, converged = _optimize(
        objective,
        initial,
        bounds,
        constraints,
        method=optimization_method,
    )

    if kind == "arch":
        final_values = np.empty(errors.size, dtype=float)
        final_values[0] = variance
        for index in range(1, errors.size):
            final_values[index] = parameters[0] + parameters[1] * errors[index - 1] ** 2
    elif kind == "garch":
        final_values = _conditional_variances(errors, parameters)
    elif kind == "gjr_garch":
        final_values = np.empty(errors.size, dtype=float)
        final_values[0] = variance
        for index in range(1, errors.size):
            indicator = 1.0 if errors[index - 1] < 0.0 else 0.0
            final_values[index] = (
                parameters[0]
                + parameters[1] * errors[index - 1] ** 2
                + parameters[2] * indicator * errors[index - 1] ** 2
                + parameters[3] * final_values[index - 1]
            )
        final_values = np.maximum(final_values, _EPSILON)
    else:
        final_values = np.empty(errors.size, dtype=float)
        final_values[0] = variance
        for index in range(1, errors.size):
            previous_std = math.sqrt(max(final_values[index - 1], _EPSILON))
            z = errors[index - 1] / previous_std
            omega, alpha, gamma, beta = parameters
            final_values[index] = math.exp(
                omega
                + beta * math.log(max(final_values[index - 1], _EPSILON))
                + alpha * (abs(z) - math.sqrt(2.0 / math.pi))
                + gamma * z
            )

    return FittedVolatilityModel(
        model=cast(VolatilityFamily, kind),
        parameters=parameters,
        log_likelihood=log_likelihood,
        observations=int(errors.size),
        converged=converged,
        annualization_factor=annualization,
        forecast_kind=kind,
        last_variance=float(final_values[-1]),
        last_error=float(errors[-1]),
    )


def fit_arch(
    returns: Sequence[float],
    annualization_factor: float = 252.0,
) -> FittedVolatilityModel:
    """Fit ARCH(1) by constrained Gaussian quasi-maximum likelihood."""
    return _garch_fit(returns, "arch", annualization_factor)


def fit_garch(
    returns: Sequence[float],
    annualization_factor: float = 252.0,
) -> FittedVolatilityModel:
    """Fit GARCH(1,1) by constrained Gaussian quasi-maximum likelihood."""
    return _garch_fit(returns, "garch", annualization_factor)


def fit_egarch(
    returns: Sequence[float],
    annualization_factor: float = 252.0,
) -> FittedVolatilityModel:
    """Fit EGARCH(1,1) with asymmetric standardized-shock response."""
    return _garch_fit(returns, "egarch", annualization_factor)


def fit_gjr_garch(
    returns: Sequence[float],
    annualization_factor: float = 252.0,
) -> FittedVolatilityModel:
    """Fit GJR-GARCH(1,1) with an explicit negative-shock term."""
    return _garch_fit(returns, "gjr_garch", annualization_factor)


def fit_stochastic_volatility(
    returns: Sequence[float],
    annualization_factor: float = 252.0,
) -> FittedVolatilityModel:
    """Fit a Gaussian quasi-likelihood log-variance AR(1) volatility model."""
    errors = _validate_returns(returns, minimum=40)
    annualization = _validate_annualization_factor(annualization_factor)
    floor = max(float(np.var(errors)) * 1e-4, _EPSILON)
    log_squared = np.log(errors * errors + floor)
    x = log_squared[:-1]
    y = log_squared[1:]
    design = np.column_stack((np.ones(x.size), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept = float(coefficients[0])
    phi = float(np.clip(coefficients[1], -0.999, 0.999))
    mu = intercept / (1.0 - phi)
    residuals = y - (intercept + phi * x)
    eta_variance = max(float(np.var(residuals)), _EPSILON)
    log_terms = np.log(2.0 * math.pi * eta_variance) + residuals * residuals / eta_variance
    log_likelihood = float(-0.5 * np.sum(log_terms))
    return FittedVolatilityModel(
        model="stochastic_volatility",
        parameters=(mu, phi, math.sqrt(eta_variance)),
        log_likelihood=log_likelihood,
        observations=int(errors.size),
        converged=True,
        annualization_factor=annualization,
        forecast_kind="stochastic_volatility",
        last_variance=float(math.exp(log_squared[-1])),
        last_error=float(errors[-1]),
    )


def fit_dcc_garch(
    returns: Sequence[Sequence[float]],
    annualization_factor: float = 252.0,
) -> FittedDCCGARCHModel:
    """Fit DCC-GARCH(1,1) using GARCH(1,1) margins and DCC dynamics."""
    matrix = np.asarray(tuple(tuple(float(value) for value in row) for row in returns), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("DCC-GARCH requires at least two return series")
    if matrix.shape[0] < 50:
        raise ValueError("DCC-GARCH requires at least 50 observations")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("DCC-GARCH returns must be finite")

    models = tuple(
        fit_garch(matrix[:, column].tolist(), annualization_factor=annualization_factor)
        for column in range(matrix.shape[1])
    )
    standardized = np.column_stack(
        tuple(
            matrix[:, column] - float(np.mean(matrix[:, column]))
            for column in range(matrix.shape[1])
        )
    )
    variances = np.column_stack(
        tuple(
            _conditional_variances(standardized[:, column], model.parameters)
            for column, model in enumerate(models)
        )
    )
    z = standardized / np.sqrt(np.maximum(variances, _EPSILON))
    q_bar = np.corrcoef(z.T)
    if not np.all(np.isfinite(q_bar)):
        raise ValueError("DCC-GARCH unconditional correlation matrix is invalid")

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        q = q_bar.copy()
        total = 0.0
        for index in range(1, z.shape[0]):
            q = (
                (1.0 - alpha - beta) * q_bar
                + alpha * np.outer(z[index - 1], z[index - 1])
                + beta * q
            )
            diagonal = np.sqrt(np.maximum(np.diag(q), _EPSILON))
            r = q / np.outer(diagonal, diagonal)
            sign, logdet = np.linalg.slogdet(r)
            if sign <= 0.0:
                return 1e12
            try:
                solved = np.linalg.solve(r, z[index])
            except np.linalg.LinAlgError:
                return 1e12
            total += logdet + float(z[index] @ solved)
        return 0.5 * total

    parameters, _objective, _converged = _optimize(
        objective,
        (0.03, 0.94),
        ((1e-8, 0.999), (0.0, 0.999)),
        ({"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1]},),
    )

    q = q_bar.copy()
    for index in range(1, z.shape[0]):
        q = (
            (1.0 - parameters[0] - parameters[1]) * q_bar
            + parameters[0] * np.outer(z[index - 1], z[index - 1])
            + parameters[1] * q
        )

    return FittedDCCGARCHModel(
        models=models,
        dcc_alpha=parameters[0],
        dcc_beta=parameters[1],
        q_bar=tuple(tuple(float(value) for value in row) for row in q_bar),
        q_last=tuple(tuple(float(value) for value in row) for row in q),
        z_last=tuple(float(value) for value in z[-1]),
    )


__all__ = [
    "FittedDCCGARCHModel",
    "FittedVolatilityModel",
    "ewma_volatility",
    "fit_arch",
    "fit_dcc_garch",
    "fit_egarch",
    "fit_garch",
    "fit_gjr_garch",
    "fit_stochastic_volatility",
    "rolling_volatility",
]
