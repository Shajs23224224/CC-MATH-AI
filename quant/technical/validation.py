from __future__ import annotations

import math
from collections.abc import Sequence


def validate_series(values: Sequence[float], minimum: int = 1) -> tuple[float, ...]:
    if minimum < 1:
        raise ValueError("minimum must be >= 1")
    checked = tuple(float(value) for value in values)
    if len(checked) < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    if any(not math.isfinite(value) for value in checked):
        raise ValueError("all observations must be finite")
    return checked


def validate_window(window: int) -> int:
    if window < 1:
        raise ValueError("window must be >= 1")
    return window


def validate_ohlcv(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...] | None]:
    checked_high = validate_series(high)
    checked_low = validate_series(low, len(checked_high))
    checked_close = validate_series(close, len(checked_high))
    if len(checked_low) != len(checked_high) or len(checked_close) != len(checked_high):
        raise ValueError("high, low and close must have equal length")
    for high_value, low_value, close_value in zip(
        checked_high, checked_low, checked_close, strict=True
    ):
        if high_value < low_value or high_value < close_value or low_value > close_value:
            raise ValueError("each OHLC row must satisfy low <= close <= high")
    checked_volume = None
    if volume is not None:
        checked_volume = validate_series(volume, len(checked_high))
        if len(checked_volume) != len(checked_high):
            raise ValueError("volume must have equal length")
        if any(value < 0 for value in checked_volume):
            raise ValueError("volume must be non-negative")
    return checked_high, checked_low, checked_close, checked_volume
