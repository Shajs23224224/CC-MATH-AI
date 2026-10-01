"""Deterministic rolling statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .validation import validate_sample


def _windows(values: Sequence[float], window: int) -> tuple[tuple[float, ...], ...]:
    if window < 2:
        raise ValueError("window must be at least 2")
    checked = validate_sample(values, minimum=window)
    return tuple(checked[index - window : index] for index in range(window, len(checked) + 1))


def rolling_mean(values: Sequence[float], window: int) -> tuple[float, ...]:
    """Return trailing-window arithmetic means."""
    return tuple(math.fsum(chunk) / window for chunk in _windows(values, window))


def rolling_std(values: Sequence[float], window: int) -> tuple[float, ...]:
    """Return trailing-window population standard deviations."""
    result = []
    for chunk in _windows(values, window):
        mean = math.fsum(chunk) / window
        result.append(math.sqrt(math.fsum((value - mean) ** 2 for value in chunk) / window))
    return tuple(result)


def rolling_zscore(values: Sequence[float], window: int) -> tuple[float, ...]:
    """Return the z-score of each window endpoint against its trailing window."""
    checked = validate_sample(values, minimum=window)
    if window < 2:
        raise ValueError("window must be at least 2")
    result = []
    for index in range(window, len(checked) + 1):
        chunk = checked[index - window : index]
        mean = math.fsum(chunk) / window
        variance = math.fsum((value - mean) ** 2 for value in chunk) / window
        std = math.sqrt(variance)
        if std == 0.0:
            raise ValueError("rolling z-score is undefined for a zero-variance window")
        result.append((chunk[-1] - mean) / std)
    return tuple(result)
