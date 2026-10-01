"""Validation shared by F14 return and performance calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence


def validate_simple_returns(returns: Sequence[float]) -> tuple[float, ...]:
    """Return finite simple returns strictly above -100%."""
    if not returns:
        raise ValueError("returns must not be empty.")
    checked = tuple(float(value) for value in returns)
    if not all(math.isfinite(value) for value in checked):
        raise ValueError("returns must contain only finite values.")
    if any(value <= -1.0 for value in checked):
        raise ValueError("simple returns must be greater than -100%.")
    return checked
