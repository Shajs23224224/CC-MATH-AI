"""Deterministic bounded scalar optimization."""

from __future__ import annotations

import math
from collections.abc import Callable

from core.contracts.math import OptimizationResult

Objective = Callable[[float], float]


def _evaluate(objective: Objective, x: float) -> float:
    value = float(objective(x))
    if not math.isfinite(value):
        raise ValueError("objective function must return finite values.")
    return value


def golden_section_minimize(
    objective: Objective,
    lower_bound: float,
    upper_bound: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
) -> OptimizationResult:
    """Minimize a unimodal scalar objective on a closed bounded interval."""
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
        raise ValueError("bounds must be finite.")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound.")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    golden_ratio_inverse = (math.sqrt(5.0) - 1.0) / 2.0
    left = float(lower_bound)
    right = float(upper_bound)

    x1 = right - golden_ratio_inverse * (right - left)
    x2 = left + golden_ratio_inverse * (right - left)
    f1 = _evaluate(objective, x1)
    f2 = _evaluate(objective, x2)

    converged = False
    iterations = 0

    while iterations < max_iterations:
        interval = right - left
        if interval <= tolerance * max(1.0, abs(left), abs(right)):
            converged = True
            break

        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + golden_ratio_inverse * (right - left)
            f2 = _evaluate(objective, x2)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - golden_ratio_inverse * (right - left)
            f1 = _evaluate(objective, x1)

        iterations += 1

    optimum = x1 if f1 <= f2 else x2
    objective_value = f1 if f1 <= f2 else f2

    return OptimizationResult(
        method="golden_section",
        optimum=optimum,
        objective_value=objective_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        iterations=iterations,
        converged=converged,
    )
