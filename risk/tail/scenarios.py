"""Versioned, deterministic portfolio stress testing for F22."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from core.contracts import StressScenario, StressTestResult


def evaluate_stress_scenario(
    position_weights: Mapping[str, float],
    scenario: StressScenario,
) -> StressTestResult:
    """Apply simple-return shocks to signed equity-relative position weights.

    Weights are exposures divided by portfolio equity: positive for long
    exposure and negative for short exposure. A held position missing a shock
    causes an error rather than being silently treated as unchanged.
    """
    if not position_weights:
        raise ValueError("position_weights must contain at least one position")

    normalized_weights: dict[str, float] = {}
    for symbol, weight in position_weights.items():
        key = symbol.strip().upper()
        if not key:
            raise ValueError("position symbols must be non-empty")
        if key in normalized_weights:
            raise ValueError(f"duplicate position symbol after normalization: {key}")
        if not math.isfinite(weight):
            raise ValueError(f"position weight for {key} must be finite")
        normalized_weights[key] = float(weight)

    missing = sorted(set(normalized_weights) - set(scenario.shocks))
    if missing:
        raise ValueError(
            "stress scenario is missing shocks for held positions: " + ", ".join(missing)
        )

    portfolio_return = sum(
        weight * scenario.shocks[symbol] for symbol, weight in normalized_weights.items()
    )
    portfolio_loss = -portfolio_return
    return StressTestResult(
        scenario_id=scenario.scenario_id,
        scenario_version=scenario.version,
        portfolio_return=float(portfolio_return),
        portfolio_loss=float(portfolio_loss),
        covered_positions=len(normalized_weights),
        methodology=(
            "Sum of signed equity-relative position weights multiplied by the versioned "
            "scenario simple-return shock for each held position."
        ),
    )


def stress_test_portfolio(
    position_weights: Mapping[str, float],
    scenarios: Sequence[StressScenario],
) -> tuple[StressTestResult, ...]:
    """Evaluate several scenarios in caller-provided order with unique identities."""
    scenario_items = tuple(scenarios)
    if not scenario_items:
        raise ValueError("at least one stress scenario is required")
    seen: set[tuple[str, str]] = set()
    results: list[StressTestResult] = []
    for scenario in scenario_items:
        identity = (scenario.scenario_id, scenario.version)
        if identity in seen:
            raise ValueError(f"duplicate stress scenario identity: {identity[0]}@{identity[1]}")
        seen.add(identity)
        results.append(evaluate_stress_scenario(position_weights, scenario))
    return tuple(results)
