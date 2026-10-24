"""Deterministic market and execution perturbations for the F41 robustness engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime

from backtest.engine import BacktestConfig, Bar
from backtest.robustness.contracts import ScenarioSpec


def _adjust_price(
    price: float,
    previous_original_close: float,
    previous_adjusted_close: float,
    scale: float,
) -> float:
    return previous_adjusted_close + (price - previous_original_close) * scale


def perturb_bars(bars: Sequence[Bar], scenario: ScenarioSpec) -> tuple[Bar, ...]:
    """Scale OHLC movement and apply one persistent price-level shock."""
    series = tuple(bars)
    if len(series) < 2:
        raise ValueError("at least two bars are required")
    volatility_adjusted: list[Bar] = [series[0]]
    for index, bar in enumerate(series[1:], start=1):
        previous_original_close = series[index - 1].close
        previous_adjusted_close = volatility_adjusted[-1].close
        scale = scenario.volatility_scale

        volatility_adjusted.append(
            Bar(
                timestamp=bar.timestamp,
                open=_adjust_price(bar.open, previous_original_close, previous_adjusted_close, scale),
                high=_adjust_price(bar.high, previous_original_close, previous_adjusted_close, scale),
                low=_adjust_price(bar.low, previous_original_close, previous_adjusted_close, scale),
                close=_adjust_price(bar.close, previous_original_close, previous_adjusted_close, scale),
                volume=bar.volume,
            )
        )

    shock_factor = 1.0 + scenario.price_shock_bps / 10_000.0
    output: list[Bar] = []
    for index, bar in enumerate(volatility_adjusted):
        factor = shock_factor if index >= scenario.shock_start_index else 1.0
        output.append(
            Bar(
                timestamp=bar.timestamp,
                open=bar.open * factor,
                high=bar.high * factor,
                low=bar.low * factor,
                close=bar.close * factor,
                volume=bar.volume,
            )
        )
    return tuple(output)


def delay_signals(
    bars: Sequence[Bar],
    targets_by_timestamp: Mapping[datetime, int],
    delay_bars: int,
) -> dict[datetime, int]:
    """Move signal observation timestamps forward, dropping signals beyond available data."""
    series = tuple(bars)
    index_by_timestamp = {bar.timestamp: index for index, bar in enumerate(series)}
    if any(timestamp not in index_by_timestamp for timestamp in targets_by_timestamp):
        raise ValueError("signals must reference an existing bar timestamp")
    delayed: dict[datetime, int] = {}
    for timestamp, target in targets_by_timestamp.items():
        target_index = index_by_timestamp[timestamp] + delay_bars
        if target_index < len(series):
            delayed[series[target_index].timestamp] = target
    return delayed


def scenario_backtest_config(
    config: BacktestConfig,
    scenario: ScenarioSpec,
) -> BacktestConfig:
    """Apply additive cost stress while preserving capital and annualization."""
    slippage = config.slippage_bps + scenario.extra_slippage_bps
    if slippage >= 10_000:
        raise ValueError("stressed total slippage must remain below 10000 bps")
    return replace(
        config,
        commission_bps=config.commission_bps + scenario.extra_commission_bps,
        slippage_bps=slippage,
    )
