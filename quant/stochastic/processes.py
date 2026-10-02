"""Deterministic mean-reversion process models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.contracts import MeanReversionFit, MeanReversionForecast


def _validate_process(values: Sequence[float]) -> np.ndarray:
    checked = np.asarray(tuple(float(value) for value in values), dtype=float)
    if checked.size < 20:
        raise ValueError("at least 20 observations are required for mean-reversion estimation")
    if not np.all(np.isfinite(checked)):
        raise ValueError("process observations must be finite")
    if float(np.var(checked)) <= 1e-12:
        raise ValueError("process observations must have positive variance")
    return checked


@dataclass(frozen=True)
class FittedMeanReversionModel:
    """Typed Ornstein-Uhlenbeck / mean-reversion estimator."""

    model: str
    observations: int
    long_run_mean: float
    speed: float
    diffusion: float
    half_life: float
    residual_std: float
    last_value: float
    dt: float

    def summary(self) -> MeanReversionFit:
        return MeanReversionFit(
            model=self.model,  # type: ignore[arg-type]
            observations=self.observations,
            long_run_mean=self.long_run_mean,
            speed=self.speed,
            diffusion=self.diffusion,
            half_life=self.half_life,
            residual_std=self.residual_std,
            converged=True,
        )

    def forecast(self, horizon: int) -> MeanReversionForecast:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        values = tuple(
            self.long_run_mean
            + (self.last_value - self.long_run_mean) * math.exp(-self.speed * self.dt * step)
            for step in range(1, horizon + 1)
        )
        return MeanReversionForecast(
            model=self.model,  # type: ignore[arg-type]
            horizon=horizon,
            values=values,
        )


def fit_ornstein_uhlenbeck(
    values: Sequence[float],
    dt: float = 1.0,
) -> FittedMeanReversionModel:
    """Estimate a continuous-time OU process from equally spaced observations."""
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    checked = _validate_process(values)
    lagged = checked[:-1]
    target = checked[1:]
    design = np.column_stack((np.ones(lagged.size), lagged))
    intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
    intercept = float(intercept)
    slope = float(slope)
    if not 0.0 < slope < 1.0:
        raise ValueError("estimated process is not mean-reverting in discrete time")
    speed = -math.log(slope) / dt
    long_run_mean = intercept / (1.0 - slope)
    residuals = target - (intercept + slope * lagged)
    residual_variance = float(np.var(residuals))
    diffusion_variance = max(residual_variance, 1e-12) * 2.0 * speed / (1.0 - slope * slope)
    diffusion = math.sqrt(diffusion_variance)
    half_life = math.log(2.0) / speed
    return FittedMeanReversionModel(
        model="ornstein_uhlenbeck",
        observations=int(checked.size),
        long_run_mean=long_run_mean,
        speed=speed,
        diffusion=diffusion,
        half_life=half_life,
        residual_std=math.sqrt(max(residual_variance, 0.0)),
        last_value=float(checked[-1]),
        dt=dt,
    )


def fit_mean_reversion(
    values: Sequence[float],
    dt: float = 1.0,
) -> FittedMeanReversionModel:
    """Estimate the same OU core under the generic mean-reversion family."""
    model = fit_ornstein_uhlenbeck(values, dt=dt)
    return FittedMeanReversionModel(
        model="mean_reversion",
        observations=model.observations,
        long_run_mean=model.long_run_mean,
        speed=model.speed,
        diffusion=model.diffusion,
        half_life=model.half_life,
        residual_std=model.residual_std,
        last_value=model.last_value,
        dt=model.dt,
    )


__all__ = [
    "FittedMeanReversionModel",
    "fit_mean_reversion",
    "fit_ornstein_uhlenbeck",
]
