"""Validated contracts for F27 probabilistic return forecasts and OOS evaluation."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ForecastingMethod = Literal["iid_empirical_bootstrap", "markov_regime_bootstrap"]


class ForecastQuantile(BaseModel):
    """One empirical quantile of a simulated cumulative simple-return distribution."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    probability: float = Field(gt=0.0, lt=1.0)
    return_value: float


class ForecastInterval(BaseModel):
    """Central predictive interval at a nominal coverage level."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    coverage: float = Field(gt=0.0, lt=1.0)
    lower_return: float
    upper_return: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ForecastInterval:
        """Reject reversed predictive interval bounds."""
        if self.lower_return > self.upper_return:
            raise ValueError("lower_return must not exceed upper_return")
        return self


class ReturnPathSimulation(BaseModel):
    """Reproducible simulated per-period simple-return paths."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model: ForecastingMethod
    observations: int = Field(ge=20)
    horizon: int = Field(ge=1)
    simulations: int = Field(ge=100)
    random_seed: int = Field(ge=0)
    simulated_returns: tuple[tuple[float, ...], ...]
    current_regime: int | None = None
    state_count: int | None = None
    transition_matrix: tuple[tuple[float, ...], ...] | None = None
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paths(self) -> ReturnPathSimulation:
        """Validate path dimensions, return domain and regime metadata."""
        if len(self.simulated_returns) != self.simulations:
            raise ValueError("simulated path count must match simulations")
        for path in self.simulated_returns:
            if len(path) != self.horizon:
                raise ValueError("each simulated path must match horizon")
            if any(not math.isfinite(value) or value <= -1.0 for value in path):
                raise ValueError("simulated simple returns must be finite and greater than -1")

        if self.model == "iid_empirical_bootstrap":
            if (
                self.current_regime is not None
                or self.state_count is not None
                or self.transition_matrix is not None
            ):
                raise ValueError("IID bootstrap must not include regime metadata")
            return self

        if self.current_regime is None or self.state_count is None:
            raise ValueError("regime bootstrap requires current_regime and state_count")
        if not 2 <= self.state_count <= 6:
            raise ValueError("state_count must be between 2 and 6")
        if not 0 <= self.current_regime < self.state_count:
            raise ValueError("current_regime must be a valid state index")
        if self.transition_matrix is None or len(self.transition_matrix) != self.state_count:
            raise ValueError("transition_matrix must match state_count")
        for row in self.transition_matrix:
            if len(row) != self.state_count:
                raise ValueError("transition_matrix must be square")
            if any(not 0.0 <= value <= 1.0 for value in row):
                raise ValueError("transition probabilities must lie in [0, 1]")
            if not math.isclose(sum(row), 1.0, abs_tol=1e-6):
                raise ValueError("each transition row must sum to one")
        return self


class ProbabilisticForecast(BaseModel):
    """Distribution summary for one cumulative-return forecast horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model: ForecastingMethod
    training_observations: int = Field(ge=20)
    horizon: int = Field(ge=1)
    simulations: int = Field(ge=100)
    random_seed: int = Field(ge=0)
    expected_return: float
    median_return: float
    predictive_std: float = Field(ge=0.0)
    probability_of_loss: float = Field(ge=0.0, le=1.0)
    quantiles: tuple[ForecastQuantile, ...] = Field(min_length=2)
    intervals: tuple[ForecastInterval, ...] = Field(min_length=1)
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distribution_summary(self) -> ProbabilisticForecast:
        """Require ordered quantiles and unique, ordered interval coverages."""
        probabilities = tuple(item.probability for item in self.quantiles)
        if tuple(sorted(set(probabilities))) != probabilities:
            raise ValueError("quantile probabilities must be unique and ascending")
        values = tuple(item.return_value for item in self.quantiles)
        if tuple(sorted(values)) != values:
            raise ValueError("quantile return values must be non-decreasing")
        if not any(math.isclose(level, 0.5, abs_tol=1e-9) for level in probabilities):
            raise ValueError("quantiles must include the median probability 0.5")

        coverages = tuple(item.coverage for item in self.intervals)
        if tuple(sorted(set(coverages))) != coverages:
            raise ValueError("interval coverages must be unique and ascending")
        for interval in self.intervals:
            if interval.lower_return > interval.upper_return:
                raise ValueError("predictive interval bounds are reversed")
        return self


class ForecastCalibrationResult(BaseModel):
    """Out-of-sample calibration and scoring metrics for one forecast horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    forecasts: int = Field(ge=1)
    horizon: int = Field(ge=1)
    nominal_coverage: float = Field(gt=0.0, lt=1.0)
    empirical_coverage: float = Field(ge=0.0, le=1.0)
    mean_interval_width: float = Field(ge=0.0)
    mean_interval_score: float = Field(ge=0.0)
    probability_of_loss_brier_score: float = Field(ge=0.0, le=1.0)
    mean_absolute_error: float = Field(ge=0.0)
    mean_squared_error: float = Field(ge=0.0)


class WalkForwardForecastEvaluation(BaseModel):
    """Chronological forecast evaluation with a stored forecast for every origin."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int = Field(ge=1)
    min_train_observations: int = Field(ge=20)
    step: int = Field(ge=1)
    origins: tuple[int, ...] = Field(min_length=1)
    forecasts: tuple[ProbabilisticForecast, ...] = Field(min_length=1)
    realized_returns: tuple[float, ...] = Field(min_length=1)
    calibration: ForecastCalibrationResult

    @model_validator(mode="after")
    def validate_evaluation(self) -> WalkForwardForecastEvaluation:
        """Ensure each model was trained strictly before its forecast origin."""
        if not (len(self.origins) == len(self.forecasts) == len(self.realized_returns)):
            raise ValueError("origins, forecasts and realized returns must have equal lengths")
        if tuple(sorted(set(self.origins))) != self.origins:
            raise ValueError("forecast origins must be unique and ascending")
        for origin, forecast in zip(self.origins, self.forecasts, strict=True):
            if origin < self.min_train_observations:
                raise ValueError("forecast origin is earlier than min_train_observations")
            if forecast.training_observations != origin:
                raise ValueError("each forecast must be trained on observations before its origin")
            if forecast.horizon != self.horizon:
                raise ValueError("all forecasts must match the evaluation horizon")
        if self.calibration.forecasts != len(self.forecasts):
            raise ValueError("calibration forecast count does not match evaluation")
        if self.calibration.horizon != self.horizon:
            raise ValueError("calibration horizon does not match evaluation")
        if any(not math.isfinite(value) or value <= -1.0 for value in self.realized_returns):
            raise ValueError("realized simple returns must be finite and greater than -1")
        return self
