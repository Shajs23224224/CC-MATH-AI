"""Validation helpers for the statistical engine."""

from __future__ import annotations

import math
from collections.abc import Sequence


def validate_sample(values: Sequence[float], *, minimum: int = 1) -> tuple[float, ...]:
    """Return finite numeric observations with an explicit minimum size."""
    if len(values) < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    checked = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in checked):
        raise ValueError("all observations must be finite")
    return checked
