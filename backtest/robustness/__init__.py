"""Public API for F41 Monte Carlo and adversarial robustness validation."""

from backtest.robustness.contracts import (
    CorrelationStressResult,
    MonteCarloConfig,
    MonteCarloSummary,
    RobustnessResult,
    ScenarioResult,
    ScenarioSpec,
)
from backtest.robustness.correlation import stress_correlation_matrix
from backtest.robustness.monte_carlo import run_monte_carlo
from backtest.robustness.runner import run_robustness

__all__ = [
    "CorrelationStressResult",
    "MonteCarloConfig",
    "MonteCarloSummary",
    "RobustnessResult",
    "ScenarioResult",
    "ScenarioSpec",
    "run_monte_carlo",
    "run_robustness",
    "stress_correlation_matrix",
]
