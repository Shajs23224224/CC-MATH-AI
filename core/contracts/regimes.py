"""Typed contracts for F26 latent regimes and online change-point diagnostics."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RegimeModel = Literal["gaussian_hmm", "markov_switching_gaussian"]
ChangeDirection = Literal["upward", "downward"]
VolatilityRegimeLabel = Literal["low", "normal", "high"]


class RegimeModelResult(BaseModel):
    """Versioned Gaussian latent-state model with filtered and smoothed probabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model: RegimeModel
    version: str = Field(min_length=1)
    observations: int = Field(ge=20)
    state_count: int = Field(ge=2, le=6)
    means: tuple[float, ...]
    variances: tuple[float, ...]
    initial_probabilities: tuple[float, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    filtered_probabilities: tuple[tuple[float, ...], ...]
    smoothed_probabilities: tuple[tuple[float, ...], ...]
    filtered_states: tuple[int, ...]
    smoothed_states: tuple[int, ...]
    log_likelihood: float
    converged: bool
    iterations: int = Field(ge=1)
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_model_output(self) -> RegimeModelResult:
        """Validate probability simplices and latent-state dimensions."""
        states = self.state_count
        if len(self.means) != states or len(self.variances) != states:
            raise ValueError("regime means and variances must match state_count")
        if tuple(sorted(self.means)) != self.means:
            raise ValueError("regime states must be ordered by ascending mean")
        if any(not math.isfinite(value) for value in (*self.means, *self.variances)):
            raise ValueError("regime parameters must be finite")
        if any(value <= 0.0 for value in self.variances):
            raise ValueError("regime variances must be positive")
        if len(self.initial_probabilities) != states:
            raise ValueError("initial probabilities must match state_count")
        if len(self.transition_matrix) != states or any(len(row) != states for row in self.transition_matrix):
            raise ValueError("transition matrix must be square and match state_count")
        for row in self.transition_matrix:
            if any(not 0.0 <= value <= 1.0 for value in row):
                raise ValueError("transition probabilities must be between zero and one")
            if not math.isclose(sum(row), 1.0, abs_tol=1e-6):
                raise ValueError("each transition matrix row must sum to one")
        if any(not 0.0 <= value <= 1.0 for value in self.initial_probabilities):
            raise ValueError("initial probabilities must be between zero and one")
        if not math.isclose(sum(self.initial_probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("initial probabilities must sum to one")

        for name, probabilities in (
            ("filtered", self.filtered_probabilities),
            ("smoothed", self.smoothed_probabilities),
        ):
            if len(probabilities) != self.observations:
                raise ValueError(f"{name} probabilities must match observation count")
            for row in probabilities:
                if len(row) != states or any(not 0.0 <= value <= 1.0 for value in row):
                    raise ValueError(f"{name} probabilities must be valid state probability rows")
                if not math.isclose(sum(row), 1.0, abs_tol=1e-5):
                    raise ValueError(f"each {name} probability row must sum to one")

        if len(self.filtered_states) != self.observations or len(self.smoothed_states) != self.observations:
            raise ValueError("regime labels must match observation count")
        if any(not 0 <= state < states for state in (*self.filtered_states, *self.smoothed_states)):
            raise ValueError("regime labels must be valid state indices")
        if not math.isfinite(self.log_likelihood):
            raise ValueError("log_likelihood must be finite")
        return self


class ChangePoint(BaseModel):
    """One detected CUSUM event at an observation index."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    index: int = Field(ge=0)
    direction: ChangeDirection
    score: float = Field(gt=0.0)


class CUSUMResult(BaseModel):
    """Sequential two-sided CUSUM path with indexed alarms."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    observations: int = Field(ge=20)
    reference_mean: float
    reference_std: float = Field(gt=0.0)
    drift: float = Field(ge=0.0)
    threshold: float = Field(gt=0.0)
    upper_cusum: tuple[float, ...]
    lower_cusum: tuple[float, ...]
    change_points: tuple[ChangePoint, ...]
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cusum_path(self) -> CUSUMResult:
        """Validate sequential CUSUM trace lengths and strictly ordered alarms."""
        if len(self.upper_cusum) != self.observations or len(self.lower_cusum) != self.observations:
            raise ValueError("CUSUM paths must match observation count")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (*self.upper_cusum, *self.lower_cusum)
        ):
            raise ValueError("CUSUM path values must be finite and non-negative")
        indices = tuple(point.index for point in self.change_points)
        if any(index >= self.observations for index in indices):
            raise ValueError("change-point indices must be inside the observed sample")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("change-point indices must be unique and strictly increasing")
        return self


class BayesianChangePointResult(BaseModel):
    """Online Bayesian run-length posterior and per-observation change probability."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    observations: int = Field(ge=20)
    hazard: float = Field(gt=0.0, lt=1.0)
    max_run_length: int = Field(ge=2)
    change_probabilities: tuple[float, ...]
    most_likely_run_lengths: tuple[int, ...]
    prior_mean: float
    prior_kappa: float = Field(gt=0.0)
    prior_alpha: float = Field(gt=0.0)
    prior_beta: float = Field(gt=0.0)
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_posterior(self) -> BayesianChangePointResult:
        """Validate online posterior range and run-length metadata."""
        if len(self.change_probabilities) != self.observations:
            raise ValueError("change probabilities must match observation count")
        if len(self.most_likely_run_lengths) != self.observations:
            raise ValueError("run lengths must match observation count")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.change_probabilities
        ):
            raise ValueError("change probabilities must be finite and between zero and one")
        if any(
            not 0 <= value < self.max_run_length for value in self.most_likely_run_lengths
        ):
            raise ValueError("most likely run lengths must be below max_run_length")
        return self


class VolatilityRegimeThresholds(BaseModel):
    """Training-calibrated trailing-volatility thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    version: str = Field(min_length=1)
    window: int = Field(ge=3)
    calibration_observations: int = Field(ge=10)
    low_quantile: float = Field(gt=0.0, lt=0.5)
    high_quantile: float = Field(gt=0.5, lt=1.0)
    low_threshold: float = Field(ge=0.0)
    high_threshold: float = Field(gt=0.0)
    unit: Literal["standard_deviation_per_period"] = "standard_deviation_per_period"

    @model_validator(mode="after")
    def validate_thresholds(self) -> VolatilityRegimeThresholds:
        """Keep quantiles ordered and thresholds strictly separated."""
        if self.low_quantile >= self.high_quantile:
            raise ValueError("low_quantile must be below high_quantile")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be below high_threshold")
        return self


class VolatilityRegimeSeries(BaseModel):
    """Regime labels from trailing realized volatility and fixed training thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    version: str = Field(min_length=1)
    observations: int = Field(ge=3)
    window: int = Field(ge=3)
    window_end_indices: tuple[int, ...] = Field(min_length=1)
    realized_volatility: tuple[float, ...]
    labels: tuple[VolatilityRegimeLabel, ...]
    thresholds: VolatilityRegimeThresholds
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_regime_series(self) -> VolatilityRegimeSeries:
        """Validate trailing windows, index positions, and deterministic labels."""
        expected_indices = tuple(range(self.window, self.observations + 1))
        if self.window > self.observations or self.window_end_indices != expected_indices:
            raise ValueError("window_end_indices must cover every trailing window with end-exclusive indices")
        if len(self.realized_volatility) != len(expected_indices) or len(self.labels) != len(expected_indices):
            raise ValueError("volatility values and labels must match trailing-window count")
        if self.thresholds.window != self.window:
            raise ValueError("volatility thresholds must use the same window length")
        if any(not math.isfinite(value) or value < 0.0 for value in self.realized_volatility):
            raise ValueError("realized volatility values must be finite and non-negative")
        for value, label in zip(self.realized_volatility, self.labels, strict=True):
            expected = (
                "low" if value < self.thresholds.low_threshold
                else "high" if value > self.thresholds.high_threshold
                else "normal"
            )
            if label != expected:
                raise ValueError("volatility labels must match their fixed thresholds")
        return self
