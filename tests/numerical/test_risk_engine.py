"""Reference and domain-invariant tests for F21 risk metrics."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.contracts import Portfolio, Position, RiskAction, RiskConstraint, RiskMetric
from quant.returns.performance import annualized_return
from risk import (
    beta,
    calmar_ratio,
    evaluate_constraints,
    evaluate_portfolio_risk,
    information_ratio,
    liquidity_risk_metrics,
    omega_ratio,
    portfolio_exposure_metrics,
    treynor_ratio,
    turnover_risk_metrics,
)


def test_beta_and_treynor_ratio_reference_values() -> None:
    benchmark = (-0.02, 0.01, 0.03, -0.01)
    returns = tuple(2.0 * value + 0.005 for value in benchmark)

    assert beta(returns, benchmark) == pytest.approx(2.0)
    assert treynor_ratio(returns, benchmark, 12.0, risk_free_per_period=0.001) == pytest.approx(
        0.054
    )


def test_calmar_ratio_uses_annualized_return_and_max_drawdown() -> None:
    returns = (0.10, -0.10, 0.05)
    assert calmar_ratio(returns, 12.0) == pytest.approx(annualized_return(returns, 12.0) / 0.10)


def test_omega_ratio_gain_to_loss_ratio() -> None:
    assert omega_ratio((0.10, -0.05, 0.02)) == pytest.approx(2.4)
    assert omega_ratio((0.10, -0.05, 0.02), threshold_per_period=0.01) == pytest.approx(5.0 / 3.0)


def test_information_ratio_uses_active_return_volatility() -> None:
    benchmark = (0.02, -0.01, 0.01, 0.03)
    returns = (0.03, -0.03, 0.02, 0.04)
    assert information_ratio(returns, benchmark, 4.0) == pytest.approx(1.0 / 3.0)


def test_portfolio_exposure_calculation_includes_short_positions() -> None:
    result = portfolio_exposure_metrics(_portfolio())

    assert result.equity == pytest.approx(1150.0)
    assert result.gross_exposure == pytest.approx(250.0)
    assert result.net_exposure == pytest.approx(150.0)
    assert result.gross_leverage == pytest.approx(250.0 / 1150.0)
    assert result.net_exposure_ratio == pytest.approx(150.0 / 1150.0)
    assert result.concentration_hhi == pytest.approx(0.68)
    assert result.largest_position_weight == pytest.approx(0.8)
    assert result.position_count == 2


def test_liquidity_and_turnover_metrics() -> None:
    liquidity = liquidity_risk_metrics(
        "aapl",
        position_notional=25_000.0,
        average_daily_dollar_volume=100_000.0,
        max_participation_rate=0.10,
    )
    assert liquidity.asset_symbol == "AAPL"
    assert liquidity.participation_rate == pytest.approx(0.25)
    assert liquidity.estimated_liquidation_days == pytest.approx(2.5)

    turnover = turnover_risk_metrics(200_000.0, 100_000.0)
    assert turnover.turnover_ratio == pytest.approx(2.0)


def test_portfolio_risk_soft_breach_reduces_exposure() -> None:
    result = evaluate_portfolio_risk(
        _portfolio(),
        (
            RiskConstraint(
                constraint_id="concentration-limit",
                name="Concentration limit",
                metric_name="concentration_hhi",
                limit=0.65,
                hard=False,
            ),
        ),
    )

    assert result.action == RiskAction.REDUCE
    assert result.reduction_factor == pytest.approx(0.5)
    assert result.triggered_constraints == ("concentration-limit",)
    assert any(metric.name == "gross_leverage" for metric in result.metrics)


def test_hard_breach_blocks_even_if_a_soft_breach_also_exists() -> None:
    metrics = (
        RiskMetric(
            name="gross_leverage",
            value=1.5,
            unit="multiple_of_equity",
            timestamp="2026-10-21T12:00:00+00:00",
            methodology="gross exposure divided by equity",
        ),
        RiskMetric(
            name="concentration_hhi",
            value=0.8,
            unit="index_0_to_1",
            timestamp="2026-10-21T12:00:00+00:00",
            methodology="sum of squared absolute weights",
        ),
    )
    result = evaluate_constraints(
        metrics,
        (
            RiskConstraint(
                constraint_id="leverage",
                name="Maximum leverage",
                metric_name="gross_leverage",
                limit=1.0,
                hard=True,
            ),
            RiskConstraint(
                constraint_id="concentration",
                name="Concentration",
                metric_name="concentration_hhi",
                limit=0.7,
                hard=False,
            ),
        ),
    )

    assert result.action == RiskAction.BLOCK
    assert result.reduction_factor == 0.0
    assert result.triggered_constraints == ("leverage", "concentration")


def test_missing_required_metric_fails_closed() -> None:
    result = evaluate_constraints(
        (),
        (
            RiskConstraint(
                constraint_id="required-liquidity",
                name="Liquidity data required",
                metric_name="liquidity_days",
                limit=3.0,
                hard=False,
            ),
        ),
    )
    assert result.action == RiskAction.BLOCK
    assert result.triggered_constraints == ("required-liquidity",)
    assert "fails closed" in result.reasons[0]


@pytest.mark.parametrize(
    ("returns", "benchmark"),
    [
        ((0.01, 0.02), (0.01,)),
        ((0.01,), (0.01,)),
    ],
)
def test_beta_rejects_misaligned_or_short_series(
    returns: tuple[float, ...],
    benchmark: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        beta(returns, benchmark)


def test_undefined_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="constant benchmark"):
        beta((0.01, 0.02, 0.03), (0.02, 0.02, 0.02))
    with pytest.raises(ValueError, match="no losses"):
        omega_ratio((0.01, 0.02))


def test_non_positive_equity_is_rejected() -> None:
    insolvent = Portfolio(
        portfolio_id="negative-equity",
        timestamp=datetime(2026, 10, 21, tzinfo=UTC),
        base_currency="USD",
        cash=0.0,
        positions=(
            Position(
                asset_symbol="SHORT",
                quantity=-2.0,
                average_price=100.0,
                market_price=100.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="equity"):
        portfolio_exposure_metrics(insolvent)


def _portfolio() -> Portfolio:
    return Portfolio(
        portfolio_id="risk-test",
        timestamp=datetime(2026, 10, 21, tzinfo=UTC),
        base_currency="USD",
        cash=1000.0,
        positions=(
            Position(
                asset_symbol="LONG",
                quantity=2.0,
                average_price=90.0,
                market_price=100.0,
            ),
            Position(
                asset_symbol="SHORT",
                quantity=-1.0,
                average_price=60.0,
                market_price=50.0,
            ),
        ),
    )
