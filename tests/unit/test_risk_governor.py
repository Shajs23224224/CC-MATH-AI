"""Behavioral tests for F38 final risk gating."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from core.contracts import RiskAction, RiskMetric
from portfolio.sizing.contracts import SizedPosition, SizingResult
from risk.governor import (
    GovernorAssetContext,
    PairwiseCorrelation,
    RiskGovernorConfig,
    RiskGovernorRequest,
    evaluate_risk_governor,
)

AS_OF = datetime(2026, 10, 23, 20, 0, tzinfo=UTC)
SYMBOLS = ("AAA", "BBB")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sizing_result(*, age_seconds: int = 30, zero_size: bool = False) -> SizingResult:
    qty_a = 0.0 if zero_size else 200.0
    qty_b = 0.0 if zero_size else 400.0
    positions = (
        SizedPosition(
            asset_symbol="AAA",
            source_target_weight=0.5,
            target_weight=0.0 if zero_size else 0.2,
            target_quantity=qty_a,
            target_notional=0.0 if zero_size else 20_000.0,
            risk_budget_amount=500.0,
            estimated_loss_at_stop=None if zero_size else 500.0,
            adv_participation=0.0 if zero_size else 0.002,
            liquidity_limited=False,
            position_weight_limited=False,
            status="ZERO_SIZE" if zero_size else "SIZED",
            rationale="synthetic F37 fixture",
        ),
        SizedPosition(
            asset_symbol="BBB",
            source_target_weight=0.5,
            target_weight=0.0 if zero_size else 0.2,
            target_quantity=qty_b,
            target_notional=0.0 if zero_size else 20_000.0,
            risk_budget_amount=500.0,
            estimated_loss_at_stop=None if zero_size else 500.0,
            adv_participation=0.0 if zero_size else 0.004,
            liquidity_limited=False,
            position_weight_limited=False,
            status="ZERO_SIZE" if zero_size else "SIZED",
            rationale="synthetic F37 fixture",
        ),
    )
    return SizingResult(
        status="ZERO_SIZE" if zero_size else "SIZED",
        method="risk_based",
        config_id="f37-test",
        config_version="1.0.0",
        allocations=positions,
        total_target_weight=0.0 if zero_size else 0.4,
        total_target_notional=0.0 if zero_size else 40_000.0,
        total_risk_budget_amount=1_000.0,
        equity=100_000.0,
        base_currency="USD",
        as_of=AS_OF - timedelta(seconds=age_seconds),
        horizon="1D",
        market_data_snapshot_id="f37-market-snapshot",
        portfolio_optimization_fingerprint=_sha("f36-fingerprint"),
        sizing_fingerprint=_sha("f37-fingerprint"),
    )


def _risk_metrics(*, evidence_age_seconds: int = 5) -> tuple[RiskMetric, ...]:
    timestamp = (AS_OF - timedelta(seconds=evidence_age_seconds)).isoformat()
    return (
        RiskMetric(
            name="max_drawdown",
            value=0.05,
            unit="fraction",
            timestamp=timestamp,
            methodology="synthetic trailing portfolio drawdown magnitude",
        ),
        RiskMetric(
            name="daily_loss_fraction",
            value=0.01,
            unit="fraction",
            timestamp=timestamp,
            methodology="synthetic intraday loss magnitude divided by equity",
        ),
        RiskMetric(
            name="max_cvar_fraction",
            value=0.04,
            unit="fraction_of_notional",
            timestamp=timestamp,
            methodology="synthetic F22 expected shortfall at declared confidence and horizon",
        ),
        RiskMetric(
            name="data_quality_score",
            value=0.99,
            unit="score_0_to_1",
            timestamp=timestamp,
            methodology="synthetic point-in-time data-quality score",
        ),
        RiskMetric(
            name="model_quality_score",
            value=0.95,
            unit="score_0_to_1",
            timestamp=timestamp,
            methodology="synthetic out-of-sample model-quality score",
        ),
    )


def _request(
    *,
    sizing_age_seconds: int = 30,
    snapshot_age_seconds: int = 5,
    evidence_age_seconds: int = 5,
    sectors: tuple[str, str] = ("TECH", "HEALTH"),
    factor_loadings: tuple[dict[str, float], dict[str, float]] = (
        {"MKT": 1.0, "VALUE": 0.1},
        {"MKT": 1.0, "VALUE": -0.1},
    ),
    correlations: tuple[PairwiseCorrelation, ...] | None = None,
    metrics: tuple[RiskMetric, ...] | None = None,
    zero_size: bool = False,
) -> RiskGovernorRequest:
    assets = (
        GovernorAssetContext(
            asset_symbol="AAA",
            sector=sectors[0],
            lot_size=100.0,
            average_daily_dollar_volume=1_000_000.0,
            current_notional=0.0,
            factor_loadings=factor_loadings[0],
        ),
        GovernorAssetContext(
            asset_symbol="BBB",
            sector=sectors[1],
            lot_size=100.0,
            average_daily_dollar_volume=1_000_000.0,
            current_notional=0.0,
            factor_loadings=factor_loadings[1],
        ),
    )
    return RiskGovernorRequest(
        sizing_result=_sizing_result(age_seconds=sizing_age_seconds, zero_size=zero_size),
        assets=assets,
        pairwise_correlations=(
            correlations
            if correlations is not None
            else (PairwiseCorrelation(asset_a="AAA", asset_b="BBB", correlation=0.40),)
        ),
        risk_metrics=metrics if metrics is not None else _risk_metrics(
            evidence_age_seconds=evidence_age_seconds
        ),
        market_data_as_of=AS_OF - timedelta(seconds=snapshot_age_seconds),
        risk_snapshot_as_of=AS_OF - timedelta(seconds=snapshot_age_seconds),
        as_of=AS_OF,
        data_snapshot_id="f38-data-snapshot",
        risk_snapshot_id="f38-risk-snapshot",
        model_snapshot_id="f38-model-snapshot",
    )


def test_all_clear_controls_allow_original_lot_sized_quantities() -> None:
    result = evaluate_risk_governor(_request())

    assert result.action == RiskAction.ALLOW
    assert result.reduction_factor == 1.0
    assert [position.approved_quantity for position in result.positions] == [200.0, 400.0]
    assert result.total_approved_notional == pytest.approx(40_000.0)
    assert result.total_approved_weight == pytest.approx(0.4)
    assert {metric.name for metric in result.metrics} >= {
        "gross_leverage",
        "concentration_hhi",
        "max_sector_weight",
        "max_abs_factor_exposure",
        "max_abs_correlation",
        "estimated_liquidation_days",
        "turnover_ratio",
    }
    assert len(result.governor_fingerprint) == 64


def test_hard_position_limit_breach_blocks_every_position() -> None:
    result = evaluate_risk_governor(
        _request(),
        RiskGovernorConfig(max_position_weight=0.15),
    )

    assert result.action == RiskAction.BLOCK
    assert result.reduction_factor == 0.0
    assert result.total_approved_notional == 0.0
    assert all(position.approved_quantity == 0.0 for position in result.positions)
    assert all(position.status == "BLOCKED" for position in result.positions)
    assert "max-position-weight" in result.triggered_constraints


def test_sector_cap_breach_is_a_hard_veto() -> None:
    result = evaluate_risk_governor(
        _request(sectors=("TECH", "TECH")),
        RiskGovernorConfig(max_sector_weight=0.35),
    )

    assert result.action == RiskAction.BLOCK
    assert "sector-weight" in result.triggered_constraints


def test_soft_correlation_breach_reduces_and_rounds_down_to_lot() -> None:
    result = evaluate_risk_governor(
        _request(),
        RiskGovernorConfig(max_abs_correlation=0.30, reduction_factor=0.50),
    )

    assert result.action == RiskAction.REDUCE
    assert result.reduction_factor == 0.50
    assert [position.approved_quantity for position in result.positions] == [100.0, 200.0]
    assert all(
        position.approved_quantity / position.lot_size
        == pytest.approx(round(position.approved_quantity / position.lot_size))
        for position in result.positions
    )
    assert all(position.approved_weight <= 0.2 for position in result.positions)
    assert "pairwise-correlation" in result.triggered_constraints


def test_missing_required_tail_metric_fails_closed() -> None:
    metrics = tuple(item for item in _risk_metrics() if item.name != "max_cvar_fraction")
    result = evaluate_risk_governor(_request(metrics=metrics))

    assert result.action == RiskAction.BLOCK
    assert result.total_approved_notional == 0.0
    assert any("max_cvar_fraction" in reason and "missing" in reason for reason in result.reasons)


def test_missing_correlation_for_active_pair_fails_closed() -> None:
    result = evaluate_risk_governor(
        _request(correlations=()),
    )

    assert result.action == RiskAction.BLOCK
    assert any("max_abs_correlation" in reason and "missing" in reason for reason in result.reasons)


def test_incomplete_factor_loadings_fail_closed() -> None:
    result = evaluate_risk_governor(
        _request(factor_loadings=({"MKT": 1.0}, {})),
    )

    assert result.action == RiskAction.BLOCK
    assert any("max_abs_factor_exposure" in reason for reason in result.reasons)


def test_drawdown_or_daily_loss_hard_breach_blocks() -> None:
    metrics = tuple(
        item.model_copy(update={"value": 0.25})
        if item.name == "max_drawdown"
        else item
        for item in _risk_metrics()
    )
    result = evaluate_risk_governor(_request(metrics=metrics))

    assert result.action == RiskAction.BLOCK
    assert "drawdown" in result.triggered_constraints


def test_stale_external_evidence_blocks_even_when_values_are_in_range() -> None:
    result = evaluate_risk_governor(
        _request(sizing_age_seconds=2, snapshot_age_seconds=2, evidence_age_seconds=20),
        RiskGovernorConfig(max_input_age_seconds=10),
    )

    assert result.action == RiskAction.BLOCK
    assert any("stale by" in reason and "risk metric" in reason for reason in result.reasons)
    assert result.total_approved_notional == 0.0


def test_stale_f37_sizing_result_blocks() -> None:
    result = evaluate_risk_governor(
        _request(sizing_age_seconds=301, snapshot_age_seconds=5),
        RiskGovernorConfig(max_input_age_seconds=300),
    )

    assert result.action == RiskAction.BLOCK
    assert any("F37 sizing timestamp is stale" in reason for reason in result.reasons)


def test_wrong_quality_metric_unit_blocks() -> None:
    metrics = tuple(
        item.model_copy(update={"unit": "ratio"})
        if item.name == "data_quality_score"
        else item
        for item in _risk_metrics()
    )
    result = evaluate_risk_governor(_request(metrics=metrics))

    assert result.action == RiskAction.BLOCK
    assert any("expected 'score_0_to_1'" in reason for reason in result.reasons)


def test_upstream_zero_size_cannot_be_turned_into_allow() -> None:
    result = evaluate_risk_governor(_request(zero_size=True))

    assert result.action == RiskAction.BLOCK
    assert result.total_approved_weight == 0.0
    assert all(position.status == "BLOCKED" for position in result.positions)


def test_governor_fingerprint_is_reproducible() -> None:
    request = _request()
    first = evaluate_risk_governor(request)
    second = evaluate_risk_governor(request)

    assert first.governor_fingerprint == second.governor_fingerprint
    assert first.positions == second.positions


def test_request_rejects_mismatched_symbols_and_lot_context() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="exactly match F37 symbol order"):
        RiskGovernorRequest.model_validate(
            {
                **request.model_dump(mode="python"),
                "assets": tuple(reversed(request.assets)),
            }
        )
    invalid_assets = (
        request.assets[0].model_copy(update={"lot_size": 150.0}),
        request.assets[1],
    )
    with pytest.raises(ValidationError, match="not a multiple"):
        RiskGovernorRequest.model_validate(
            {**request.model_dump(mode="python"), "assets": invalid_assets}
        )
