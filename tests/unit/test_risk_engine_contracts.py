"Contract and safety tests for the F21 risk engine."

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.contracts import (
    LiquidityRiskMetrics,
    PortfolioExposureMetrics,
    RiskAction,
    RiskConstraint,
    RiskEvaluation,
    RiskMetric,
    TurnoverMetrics,
)
from risk import evaluate_constraints, liquidity_risk_metrics, turnover_risk_metrics


def test_risk_constraint_supports_upper_and_lower_bounds() -> None:
    maximum = RiskConstraint(
        constraint_id="max-drawdown",
        name="Maximum drawdown",
        metric_name="drawdown",
        limit=0.2,
        operator="max",
    )
    minimum = RiskConstraint(
        constraint_id="min-liquidity",
        name="Minimum liquidity score",
        metric_name="liquidity_score",
        limit=0.4,
        operator="min",
    )
    assert maximum.operator == "max"
    assert minimum.operator == "min"


def test_risk_metric_and_evaluation_are_typed_and_immutable() -> None:
    metric = RiskMetric(
        name="gross_leverage",
        value=0.8,
        unit="multiple_of_equity",
        timestamp="2026-10-21T12:00:00+00:00",
        methodology="gross exposure divided by equity",
    )
    evaluation = RiskEvaluation(
        action=RiskAction.ALLOW,
        metrics=(metric,),
        reduction_factor=1.0,
    )

    with pytest.raises(ValidationError):
        metric.value = 2.0
    with pytest.raises(ValidationError):
        RiskMetric(
            name="invalid",
            value=float("nan"),
            unit="ratio",
            timestamp="now",
            methodology="invalid input test",
        )
    assert evaluation.action == RiskAction.ALLOW


def test_liquidity_and_turnover_contracts_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        LiquidityRiskMetrics(
            asset_symbol="AAPL",
            position_notional=100.0,
            average_daily_dollar_volume=0.0,
            max_participation_rate=0.1,
            participation_rate=0.1,
            estimated_liquidation_days=10.0,
        )
    with pytest.raises(ValidationError):
        TurnoverMetrics(
            traded_notional=100.0,
            average_portfolio_value=0.0,
            turnover_ratio=1.0,
        )
    with pytest.raises(ValueError):
        liquidity_risk_metrics("AAPL", 100.0, 1000.0, max_participation_rate=0.0)
    with pytest.raises(ValueError):
        turnover_risk_metrics(100.0, 0.0)


def test_duplicate_metrics_and_constraints_are_rejected() -> None:
    metric = RiskMetric(
        name="drawdown",
        value=0.1,
        unit="fraction",
        timestamp="2026-10-21T12:00:00+00:00",
        methodology="reference",
    )
    constraint = RiskConstraint(
        constraint_id="drawdown-limit",
        name="Drawdown limit",
        metric_name="drawdown",
        limit=0.2,
    )
    with pytest.raises(ValueError, match="duplicate risk metric"):
        evaluate_constraints((metric, metric), (constraint,))
    with pytest.raises(ValueError, match="duplicate risk constraint"):
        evaluate_constraints((metric,), (constraint, constraint))


def test_invalid_reduction_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="reduction_factor"):
        evaluate_constraints((), (), reduction_factor=1.0)


def test_missing_metric_blocks_even_for_soft_constraint() -> None:
    result = evaluate_constraints(
        (),
        (
            RiskConstraint(
                constraint_id="required",
                name="Required data",
                metric_name="required_metric",
                limit=0.5,
                hard=False,
            ),
        ),
    )
    assert result.action == RiskAction.BLOCK
    assert result.reduction_factor == 0.0



def test_risk_evaluation_enforces_action_factor_invariants() -> None:
    metric = RiskMetric(
        name="drawdown",
        value=0.1,
        unit="fraction",
        timestamp="2026-10-21T12:00:00+00:00",
        methodology="test",
    )
    with pytest.raises(ValidationError, match="ALLOW requires"):
        RiskEvaluation(action=RiskAction.ALLOW, metrics=(metric,), reduction_factor=0.5)
    with pytest.raises(ValidationError, match="BLOCK requires"):
        RiskEvaluation(action=RiskAction.BLOCK, metrics=(metric,), reduction_factor=1.0)
    with pytest.raises(ValidationError, match="REDUCE requires"):
        RiskEvaluation(action=RiskAction.REDUCE, metrics=(metric,), reduction_factor=1.0)


def test_empty_constraints_fail_closed() -> None:
    result = evaluate_constraints((), ())
    assert result.action == RiskAction.BLOCK
    assert result.reduction_factor == 0.0
    assert "no risk constraints configured" in result.reasons[0]


def test_metric_unit_mismatch_blocks_even_for_soft_constraint() -> None:
    metric = RiskMetric(
        name="drawdown",
        value=0.2,
        unit="percent",
        timestamp="2026-10-21T12:00:00+00:00",
        methodology="percentage points",
    )
    constraint = RiskConstraint(
        constraint_id="drawdown",
        name="Maximum drawdown",
        metric_name="drawdown",
        limit=0.25,
        hard=False,
        expected_unit="fraction",
    )
    result = evaluate_constraints((metric,), (constraint,))
    assert result.action == RiskAction.BLOCK
    assert result.reduction_factor == 0.0
    assert "expected 'fraction'" in result.reasons[0]


def test_canonical_drawdown_unit_is_checked_without_explicit_expected_unit() -> None:
    metric = RiskMetric(
        name="drawdown",
        value=20.0,
        unit="percent",
        timestamp="2026-10-21T12:00:00+00:00",
        methodology="percentage points",
    )
    constraint = RiskConstraint(
        constraint_id="drawdown",
        name="Maximum drawdown",
        metric_name="drawdown",
        limit=0.25,
        hard=False,
    )
    result = evaluate_constraints((metric,), (constraint,))
    assert result.action == RiskAction.BLOCK
    assert "expected 'fraction'" in result.reasons[0]


def test_matching_metric_unit_allows_constraint_evaluation() -> None:
    metric = RiskMetric(
        name="drawdown",
        value=0.2,
        unit="fraction",
        timestamp="2026-10-21T12:00:00+00:00",
        methodology="fractional drawdown",
    )
    constraint = RiskConstraint(
        constraint_id="drawdown",
        name="Maximum drawdown",
        metric_name="drawdown",
        limit=0.25,
        expected_unit="fraction",
    )
    result = evaluate_constraints((metric,), (constraint,))
    assert result.action == RiskAction.ALLOW
    assert result.reduction_factor == 1.0


def test_exposure_contract_fields_are_bounded() -> None:
    with pytest.raises(ValidationError):
        PortfolioExposureMetrics(
            portfolio_id="invalid",
            timestamp="2026-10-21T12:00:00+00:00",
            base_currency="USD",
            equity=1000.0,
            gross_exposure=200.0,
            net_exposure=200.0,
            gross_leverage=0.2,
            net_exposure_ratio=0.2,
            cash_weight=0.8,
            concentration_hhi=1.2,
            largest_position_weight=1.0,
            position_count=1,
        )
