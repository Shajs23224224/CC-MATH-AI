from datetime import UTC, datetime, timedelta

import pytest

from backtest.engine import BacktestConfig, Bar, run_backtest
from backtest.robustness import (
    MonteCarloConfig,
    ScenarioSpec,
    run_monte_carlo,
    run_robustness,
    stress_correlation_matrix,
)


def bars(*closes: float) -> list[Bar]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        Bar(start + timedelta(days=index), close, close, close, close, 1_000.0)
        for index, close in enumerate(closes)
    ]


def test_monte_carlo_is_reproducible_for_fixed_seed_and_configuration() -> None:
    series = bars(100, 101, 98, 102, 99, 104, 103, 108)
    baseline = run_backtest(
        series,
        {series[0].timestamp: 1, series[5].timestamp: 0},
        BacktestConfig(commission_bps=0, slippage_bps=0),
    )
    config = MonteCarloConfig(simulations=250, seed=91, method="moving_block", block_size=3)
    first = run_monte_carlo(baseline, config)
    second = run_monte_carlo(baseline, config)
    assert first == second
    assert first.simulations == 250
    assert 0 <= first.probability_of_loss <= 1
    assert 0 <= first.p95_max_drawdown <= 1
    assert first.lower_quantile_return <= first.upper_quantile_return


def test_iid_and_moving_block_are_supported_and_seed_is_recorded() -> None:
    series = bars(100, 101, 99, 102, 98, 103)
    baseline = run_backtest(series, {series[0].timestamp: 1})
    result = run_monte_carlo(baseline, MonteCarloConfig(simulations=25, seed=7, method="iid"))
    assert result.method == "iid"
    assert result.seed == 7


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"simulations": 0}, "simulations"),
        ({"seed": -1}, "seed"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"method": "unknown"}, "method"),
        ({"block_size": 0}, "block_size"),
    ],
)
def test_monte_carlo_config_rejects_invalid_assumptions(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MonteCarloConfig(**kwargs)


def test_monte_carlo_rejects_non_positive_equity_curves() -> None:
    series = bars(100, 101, 102)
    baseline = run_backtest(series, {series[0].timestamp: 1})
    invalid = baseline.__class__(
        **{
            **{field: getattr(baseline, field) for field in baseline.__dataclass_fields__},
            "equity_curve": tuple(
                point.__class__(point.timestamp, point.cash, point.units, point.close, -1.0)
                for point in baseline.equity_curve
            ),
        }
    )
    with pytest.raises(ValueError, match="strictly positive"):
        run_monte_carlo(invalid, MonteCarloConfig(simulations=2))


def test_adversarial_price_shock_reduces_long_strategy_return() -> None:
    series = bars(100, 100, 100, 100, 100)
    result = run_robustness(
        series,
        {series[0].timestamp: 1},
        backtest_config=BacktestConfig(commission_bps=0, slippage_bps=0),
        monte_carlo_config=MonteCarloConfig(simulations=50, seed=2),
        scenarios=(ScenarioSpec(name="gap-down", price_shock_bps=-2_000, shock_start_index=2),),
    )
    assert result.scenarios[0].total_return < result.baseline_return
    assert result.scenarios[0].return_delta < 0
    assert result.stress_degradation_fraction == 1.0
    assert result.fragility_score == pytest.approx(
        0.5 * result.monte_carlo.probability_of_loss + 0.5
    )
    assert len(result.fingerprint) == 64


def test_higher_costs_and_delayed_execution_are_applied_in_scenario() -> None:
    series = bars(100, 110, 120, 130, 140, 150)
    result = run_robustness(
        series,
        {series[0].timestamp: 2, series[2].timestamp: 0},
        backtest_config=BacktestConfig(commission_bps=0, slippage_bps=0),
        monte_carlo_config=MonteCarloConfig(simulations=10, seed=1),
        scenarios=(
            ScenarioSpec(
                name="friction-and-latency",
                extra_commission_bps=30,
                extra_slippage_bps=20,
                execution_delay_bars=1,
            ),
        ),
    )
    assert result.scenarios[0].trades >= 1
    assert result.scenarios[0].commissions > 0
    assert result.scenarios[0].slippage_costs > 0


def test_robustness_fingerprint_and_scenario_validation() -> None:
    series = bars(100, 105, 102, 110, 107)
    args = {
        "bars": series,
        "targets_by_timestamp": {series[0].timestamp: 1},
        "monte_carlo_config": MonteCarloConfig(simulations=40, seed=17),
        "scenarios": (ScenarioSpec(name="volatility-x2", volatility_scale=2.0),),
    }
    assert run_robustness(**args).fingerprint == run_robustness(**args).fingerprint
    with pytest.raises(ValueError, match="unique"):
        run_robustness(
            series,
            {},
            monte_carlo_config=MonteCarloConfig(simulations=2),
            scenarios=(ScenarioSpec(name="same"), ScenarioSpec(name="same")),
        )


def test_correlation_stress_moves_matrix_to_declared_adverse_target() -> None:
    result = stress_correlation_matrix(
        {
            "asset_a": (-0.03, -0.01, 0.01, 0.03, 0.02),
            "asset_b": (-0.02, -0.01, 0.02, 0.04, 0.01),
        },
        shock=-0.75,
    )
    baseline_corr = result.baseline_matrix[0][1]
    stressed_corr = result.stressed_matrix[0][1]
    assert baseline_corr > 0
    assert stressed_corr < baseline_corr
    assert result.stressed_matrix[0][0] == 1.0
    assert len(result.fingerprint) == 64


def test_correlation_stress_validates_alignment_variance_and_shock() -> None:
    with pytest.raises(ValueError, match="at least two"):
        stress_correlation_matrix({"only": (0.1, 0.2)}, 0.2)
    with pytest.raises(ValueError, match="equal lengths"):
        stress_correlation_matrix({"a": (0.1, 0.2), "b": (0.2, 0.3, 0.1)}, 0.0)
    with pytest.raises(ValueError, match="constant"):
        stress_correlation_matrix({"a": (0.1, 0.1), "b": (0.2, 0.3)}, 0.0)
    with pytest.raises(ValueError, match="correlation shock"):
        stress_correlation_matrix({"a": (0.1, 0.2), "b": (0.2, 0.4)}, 1.1)


def test_correlation_shocks_are_included_in_overall_audit() -> None:
    series = bars(100, 101, 102, 104)
    result = run_robustness(
        series,
        {series[0].timestamp: 1},
        monte_carlo_config=MonteCarloConfig(simulations=3),
        correlation_returns_by_asset={"a": (-0.1, 0.0, 0.1), "b": (0.1, 0.0, -0.1)},
        correlation_shocks=(-0.5, 0.5),
    )
    assert len(result.correlation_stresses) == 2
    assert result.correlation_stresses[0].shock == -0.5
    with pytest.raises(ValueError, match="required"):
        run_robustness(
            series,
            {},
            monte_carlo_config=MonteCarloConfig(simulations=2),
            correlation_shocks=(0.3,),
        )
