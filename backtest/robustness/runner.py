"""Orchestration and audit fingerprinting for F41 robustness experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from math import isfinite

from backtest.engine import BacktestConfig, BacktestResult, Bar, run_backtest
from backtest.robustness.contracts import (
    CorrelationStressResult,
    MonteCarloConfig,
    RobustnessResult,
    ScenarioResult,
    ScenarioSpec,
)
from backtest.robustness.correlation import stress_correlation_matrix
from backtest.robustness.monte_carlo import run_monte_carlo
from backtest.robustness.scenarios import delay_signals, perturb_bars, scenario_backtest_config


def _result_metrics(result: BacktestResult) -> dict[str, float | int]:
    return {
        "final_equity": result.final_equity,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "trades": result.trades,
        "commissions": result.commissions,
        "slippage_costs": result.slippage_costs,
    }


def _bars_payload(bars: tuple[Bar, ...]) -> list[dict[str, float | str]]:
    return [
        {
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars
    ]


def run_robustness(
    bars: Sequence[Bar],
    targets_by_timestamp: Mapping[datetime, int],
    *,
    backtest_config: BacktestConfig | None = None,
    monte_carlo_config: MonteCarloConfig | None = None,
    scenarios: Sequence[ScenarioSpec] = (),
    correlation_returns_by_asset: Mapping[str, Sequence[float]] | None = None,
    correlation_shocks: Sequence[float] = (),
) -> RobustnessResult:
    """Run baseline, bootstrap and explicit adverse scenarios with reproducible metadata."""
    series = tuple(bars)
    settings = backtest_config or BacktestConfig()
    mc_settings = monte_carlo_config or MonteCarloConfig()
    baseline = run_backtest(series, targets_by_timestamp, settings)
    monte_carlo = run_monte_carlo(baseline, mc_settings)

    scenario_names: set[str] = set()
    scenario_results: list[ScenarioResult] = []
    for scenario in scenarios:
        if not isinstance(scenario, ScenarioSpec):
            raise ValueError("scenarios must contain only ScenarioSpec values")
        if scenario.name in scenario_names:
            raise ValueError("scenario names must be unique")
        scenario_names.add(scenario.name)
        if scenario.price_shock_bps != 0 and scenario.shock_start_index >= len(series):
            raise ValueError("shock_start_index must refer to an existing bar")
        stressed_bars = perturb_bars(series, scenario)
        stressed_signals = delay_signals(
            series, targets_by_timestamp, scenario.execution_delay_bars
        )
        stressed_config = scenario_backtest_config(settings, scenario)
        stressed = run_backtest(stressed_bars, stressed_signals, stressed_config)
        scenario_results.append(
            ScenarioResult(
                name=scenario.name,
                spec=scenario,
                total_return=stressed.total_return,
                max_drawdown=stressed.max_drawdown,
                sharpe_ratio=stressed.sharpe_ratio,
                trades=stressed.trades,
                commissions=stressed.commissions,
                slippage_costs=stressed.slippage_costs,
                return_delta=stressed.total_return - baseline.total_return,
                drawdown_delta=stressed.max_drawdown - baseline.max_drawdown,
            )
        )

    if correlation_shocks and correlation_returns_by_asset is None:
        raise ValueError("correlation_returns_by_asset is required for correlation_shocks")
    correlation_results: list[CorrelationStressResult] = []
    for shock in correlation_shocks:
        correlation_results.append(
            stress_correlation_matrix(correlation_returns_by_asset or {}, shock)
        )

    degradation_fraction = (
        sum(item.total_return < baseline.total_return for item in scenario_results)
        / len(scenario_results)
        if scenario_results
        else 0.0
    )
    # A transparent heuristic, not a calibrated probability or risk limit.
    fragility_score = 0.5 * monte_carlo.probability_of_loss + 0.5 * degradation_fraction
    if not isfinite(fragility_score):
        raise ArithmeticError("fragility score must be finite")

    payload = {
        "bars": _bars_payload(series),
        "targets": [
            {"timestamp": timestamp.isoformat(), "target": target}
            for timestamp, target in sorted(targets_by_timestamp.items())
        ],
        "backtest_config": asdict(settings),
        "monte_carlo_config": asdict(mc_settings),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "correlation_returns_by_asset": {
            name: tuple(values)
            for name, values in sorted((correlation_returns_by_asset or {}).items())
        },
        "correlation_shocks": tuple(correlation_shocks),
        "baseline_metrics": _result_metrics(baseline),
        "monte_carlo": asdict(monte_carlo),
        "scenario_results": [asdict(item) for item in scenario_results],
        "correlation_results": [
            {
                "asset_names": item.asset_names,
                "observations": item.observations,
                "shock": item.shock,
                "baseline_matrix": item.baseline_matrix,
                "stressed_matrix": item.stressed_matrix,
                "fingerprint": item.fingerprint,
            }
            for item in correlation_results
        ],
        "stress_degradation_fraction": degradation_fraction,
        "fragility_score": fragility_score,
    }
    fingerprint = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return RobustnessResult(
        baseline_return=baseline.total_return,
        baseline_max_drawdown=baseline.max_drawdown,
        monte_carlo=monte_carlo,
        scenarios=tuple(scenario_results),
        correlation_stresses=tuple(correlation_results),
        stress_degradation_fraction=degradation_fraction,
        fragility_score=fragility_score,
        fingerprint=fingerprint,
    )
