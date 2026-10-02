"""Validation helpers for temporal models."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.contracts import TemporalSplit


def validate_univariate(values: Sequence[float], minimum: int = 8) -> tuple[float, ...]:
    """Validate a finite univariate time series."""
    checked = tuple(float(value) for value in values)
    if len(checked) < minimum:
        raise ValueError(f"time series is below the minimum of {minimum} observations")
    if not all(math.isfinite(value) for value in checked):
        raise ValueError("time-series values must be finite")
    return checked


def validate_matrix(
    values: Sequence[Sequence[float]],
    minimum_rows: int = 10,
    minimum_columns: int = 2,
) -> tuple[tuple[float, ...], ...]:
    """Validate a finite rectangular multivariate time series."""
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if not rows:
        raise ValueError("matrix must not be empty")
    if len(rows) < minimum_rows:
        raise ValueError(f"matrix must contain at least {minimum_rows} observations")
    width = len(rows[0])
    if width < minimum_columns:
        raise ValueError(f"matrix must contain at least {minimum_columns} series")
    if any(len(row) != width for row in rows):
        raise ValueError("matrix must be rectangular")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("matrix values must be finite")
    return rows


def validate_exogenous(
    values: Sequence[Sequence[float]],
    observations: int,
    columns: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Validate exogenous regressors aligned one row per observation."""
    checked = validate_matrix(values, minimum_rows=1, minimum_columns=1)
    if len(checked) != observations:
        raise ValueError("exogenous observations must match the target series")
    if columns is not None and len(checked[0]) != columns:
        raise ValueError("exogenous column count does not match the fitted model")
    return checked


def temporal_train_test_split(
    values: Sequence[float],
    train_size: int,
) -> tuple[tuple[float, ...], tuple[float, ...], TemporalSplit]:
    """Split strictly in chronological order; no shuffling is permitted."""
    checked = validate_univariate(values, minimum=2)
    if not 1 <= train_size < len(checked):
        raise ValueError("train_size must leave at least one test observation")
    train = checked[:train_size]
    test = checked[train_size:]
    split = TemporalSplit(
        train_size=len(train),
        test_size=len(test),
        total_size=len(checked),
    )
    return train, test, split


def validate_order(order: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate ARIMA-style non-negative orders."""
    if len(order) != 3 or any(not isinstance(value, int) for value in order):
        raise ValueError("order must be a triple of integers")
    if any(value < 0 for value in order):
        raise ValueError("ARIMA orders must be non-negative")
    if sum(order) == 0:
        raise ValueError("at least one ARIMA order must be positive")
    return order


def validate_seasonal_order(
    order: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Validate SARIMA seasonal orders."""
    if len(order) != 4 or any(not isinstance(value, int) for value in order):
        raise ValueError("seasonal_order must contain four integers")
    if any(value < 0 for value in order):
        raise ValueError("seasonal orders must be non-negative")
    if order[3] < 2:
        raise ValueError("seasonal period must be at least 2")
    return order


def validate_lag(lag: int) -> int:
    """Validate a positive lag/order."""
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("lag must be a positive integer")
    return lag


def validate_forecast_horizon(horizon: int) -> int:
    """Validate a positive forecast horizon."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("forecast horizon must be a positive integer")
    return horizon
