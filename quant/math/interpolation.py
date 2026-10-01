"""Deterministic interpolation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def linear_interpolate(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x: float,
) -> float:
    """Linearly interpolate or extrapolate a single point."""
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1, x)):
        raise ValueError("all inputs must be finite.")
    if x0 == x1:
        raise ValueError("x0 and x1 must be different.")
    fraction = (x - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)


def linear_interpolate_series(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x: float,
) -> float:
    """Interpolate piecewise-linearly over strictly increasing x coordinates."""
    if not x_values or not y_values:
        raise ValueError("x_values and y_values must not be empty.")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if len(x_values) < 2:
        raise ValueError("at least two interpolation points are required.")
    if not math.isfinite(x):
        raise ValueError("x must be finite.")

    checked_x = tuple(float(value) for value in x_values)
    checked_y = tuple(float(value) for value in y_values)
    if not all(math.isfinite(value) for value in (*checked_x, *checked_y)):
        raise ValueError("all interpolation coordinates must be finite.")
    if any(left >= right for left, right in zip(checked_x, checked_x[1:])):
        raise ValueError("x_values must be strictly increasing.")

    if x <= checked_x[0]:
        return linear_interpolate(checked_x[0], checked_y[0], checked_x[1], checked_y[1], x)
    if x >= checked_x[-1]:
        return linear_interpolate(checked_x[-2], checked_y[-2], checked_x[-1], checked_y[-1], x)

    for index in range(1, len(checked_x)):
        if x <= checked_x[index]:
            return linear_interpolate(
                checked_x[index - 1],
                checked_y[index - 1],
                checked_x[index],
                checked_y[index],
                x,
            )

    raise RuntimeError("unreachable interpolation state.")
