"""Immutable, validated contracts for F41 robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

SamplingMethod = Literal["iid", "moving_block"]


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Versionable assumptions for bootstrapping observed strategy-period returns."""

    simulations: int = 1_000
    seed: int = 42
    confidence_level: float = 0.95
    method: SamplingMethod = "moving_block"
    block_size: int = 5

    def __post_init__(self) -> None:
        if (
            not isinstance(self.simulations, int)
            or isinstance(self.simulations, bool)
            or not 1 <= self.simulations <= 100_000
        ):
            raise ValueError("simulations must be an integer between 1 and 100000")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isfinite(self.confidence_level) or not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be finite and between 0.5 and 1")
        if self.method not in ("iid", "moving_block"):
            raise ValueError("method must be iid or moving_block")
        if (
            not isinstance(self.block_size, int)
            or isinstance(self.block_size, bool)
            or self.block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """A deterministic adverse perturbation applied to bars, costs, or signal latency."""

    name: str
    volatility_scale: float = 1.0
    price_shock_bps: float = 0.0
    shock_start_index: int = 1
    extra_commission_bps: float = 0.0
    extra_slippage_bps: float = 0.0
    execution_delay_bars: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be a non-empty string")
        if not isfinite(self.volatility_scale) or not 0 < self.volatility_scale <= 5:
            raise ValueError("volatility_scale must be finite and in (0, 5]")
        if (
            not isfinite(self.price_shock_bps)
            or self.price_shock_bps <= -10_000
            or abs(self.price_shock_bps) > 10_000
        ):
            raise ValueError("price_shock_bps must be finite and in [-10000, 10000)")
        if (
            not isinstance(self.shock_start_index, int)
            or isinstance(self.shock_start_index, bool)
            or self.shock_start_index < 1
        ):
            raise ValueError("shock_start_index must be an integer >= 1")
        for name, value in (
            ("extra_commission_bps", self.extra_commission_bps),
            ("extra_slippage_bps", self.extra_slippage_bps),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not isinstance(self.execution_delay_bars, int)
            or isinstance(self.execution_delay_bars, bool)
            or self.execution_delay_bars < 0
        ):
            raise ValueError("execution_delay_bars must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class MonteCarloSummary:
    """Distribution of simulated terminal returns and drawdowns."""

    simulations: int
    seed: int
    method: SamplingMethod
    block_size: int
    confidence_level: float
    mean_return: float
    median_return: float
    lower_quantile_return: float
    upper_quantile_return: float
    probability_of_loss: float
    expected_shortfall: float
    p95_max_drawdown: float


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Backtest diagnostics for a single declared adversarial scenario."""

    name: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: int
    commissions: float
    slippage_costs: float
    return_delta: float
    drawdown_delta: float


@dataclass(frozen=True, slots=True)
class CorrelationStressResult:
    """Observed and stressed Pearson correlation matrices over aligned returns."""

    asset_names: tuple[str, ...]
    observations: int
    shock: float
    baseline_matrix: tuple[tuple[float, ...], ...]
    stressed_matrix: tuple[tuple[float, ...], ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    """Baseline, Monte Carlo, deterministic stress scenarios and reproducibility audit."""

    baseline_return: float
    baseline_max_drawdown: float
    monte_carlo: MonteCarloSummary
    scenarios: tuple[ScenarioResult, ...]
    correlation_stresses: tuple[CorrelationStressResult, ...]
    stress_degradation_fraction: float
    fragility_score: float
    fingerprint: str
