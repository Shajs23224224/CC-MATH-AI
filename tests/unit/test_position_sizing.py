"""Behavioral and numerical tests for F37 position sizing."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from core.contracts import DecisionRecord, DependenceMatrix, SignalType
from portfolio.optimization import (
    PortfolioAssetInput,
    PortfolioOptimizationConfig,
    PortfolioOptimizationRequest,
    optimize_portfolio,
)
from portfolio.sizing import (
    PositionSizingConfig,
    PositionSizingRequest,
    SizingAssetInput,
    size_positions,
)
from signals.decision import DecisionOutput, DecisionProbabilities

AS_OF = datetime(2026, 10, 23, 15, 0, tzinfo=UTC)
SYMBOLS = ("AAA", "BBB")
RETURNS = (0.12, 0.08)
COVARIANCE = ((0.04, 0.006), (0.006, 0.01))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decision(symbol: str, expected_return: float) -> DecisionOutput:
    probabilities = DecisionProbabilities(buy=0.8, hold=0.1, sell=0.1)
    record = DecisionRecord(
        decision_id=f"decision-{symbol}",
        asset_symbol=symbol,
        timestamp=AS_OF - timedelta(minutes=10),
        signal=SignalType.BUY,
        signal_probability=probabilities.buy,
        model_confidence=0.8,
        expected_return=expected_return,
        expected_risk=0.1,
        risk_adjusted_score=expected_return / 0.1,
        recommended_horizon="1D",
        suggested_position_size=0.0,
        regime="SYNTHETIC",
        data_snapshot_id="f37-synthetic-f35",
        model_versions=("synthetic-model@1.0.0",),
        feature_versions=("synthetic-features@1.0.0",),
    )
    return DecisionOutput(
        record=record,
        probabilities=probabilities,
        status="DECIDED",
        abstained=False,
        abstention_reason=None,
        agreement=0.9,
        uncertainty=0.1,
        evidence=(),
        constraints=(),
        decision_config_id="test-decision",
        decision_config_version="1.0.0",
        probability_calibration_status="validated_oos",
        ensemble_fingerprint=_sha(f"ensemble-{symbol}"),
        decision_fingerprint=_sha(f"decision-{symbol}"),
    )


def _optimization():
    assets = tuple(
        PortfolioAssetInput(
            decision=_decision(symbol, expected_return),
            current_weight=0.0,
            minimum_weight=0.0,
            maximum_weight=1.0,
            sector="TEST",
            eligible=True,
        )
        for symbol, expected_return in zip(SYMBOLS, RETURNS, strict=True)
    )
    covariance = DependenceMatrix(
        assets=SYMBOLS,
        method="sample_covariance",
        kind="covariance",
        observations=60,
        start_index=0,
        end_index=60,
        matrix=COVARIANCE,
    )
    request = PortfolioOptimizationRequest(
        assets=assets,
        covariance=covariance,
        as_of=AS_OF,
        horizon="1D",
        covariance_as_of=AS_OF - timedelta(hours=1),
        covariance_horizon="1D",
        data_snapshot_id="f37-decision-snapshot",
        covariance_data_snapshot_id="f37-covariance-snapshot",
    )
    result = optimize_portfolio(
        request,
        PortfolioOptimizationConfig(method="equal_weight"),
    )
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    return result


def _request(
    *,
    stops: tuple[float | None, ...] = (95.0, 45.0),
    volumes: tuple[float, ...] = (100_000.0, 100_000.0),
    market_data_as_of: datetime = AS_OF - timedelta(minutes=1),
    equity: float = 100_000.0,
) -> PositionSizingRequest:
    assets = tuple(
        SizingAssetInput(
            asset_symbol=symbol,
            price=price,
            expected_return=expected_return,
            volatility=volatility,
            confidence=confidence,
            average_daily_volume=volume,
            lot_size=1.0,
            stop_loss_price=stop,
        )
        for symbol, price, expected_return, volatility, confidence, volume, stop in zip(
            SYMBOLS,
            (100.0, 50.0),
            RETURNS,
            (0.20, 0.10),
            (1.0, 0.5),
            volumes,
            stops,
            strict=True,
        )
    )
    return PositionSizingRequest(
        portfolio_optimization=_optimization(),
        assets=assets,
        equity=equity,
        base_currency="cop",
        as_of=AS_OF,
        market_data_as_of=market_data_as_of,
        horizon="1D",
        market_data_snapshot_id="f37-market-snapshot",
    )


@pytest.mark.parametrize(
    "method",
    (
        "fixed_fractional",
        "volatility_based",
        "risk_based",
        "kelly",
        "fractional_kelly",
        "confidence_weighting",
    ),
)
def test_all_six_methods_return_bounded_auditable_quantities(method: str) -> None:
    request = _request()
    result = size_positions(request, PositionSizingConfig(method=method))

    assert result.status in {"SIZED", "PARTIAL", "ZERO_SIZE"}
    assert result.method == method
    assert len(result.allocations) == 2
    assert all(item.target_quantity >= 0.0 for item in result.allocations)
    assert all(
        item.target_weight <= item.source_target_weight + 1e-8
        for item in result.allocations
    )
    assert result.total_target_weight <= 1.0
    assert result.total_target_notional == pytest.approx(
        sum(item.target_notional for item in result.allocations)
    )
    assert len(result.sizing_fingerprint) == 64
    assert result.portfolio_optimization_fingerprint == request.portfolio_optimization.optimization_fingerprint


def test_risk_based_method_enforces_stop_loss_risk_budget() -> None:
    result = size_positions(_request(), PositionSizingConfig(method="risk_based"))

    assert result.status == "SIZED"
    assert sum(item.estimated_loss_at_stop for item in result.allocations) <= (
        result.total_risk_budget_amount + 1e-6
    )
    assert result.total_risk_budget_amount == pytest.approx(1_000.0)
    assert all(item.estimated_loss_at_stop > 0.0 for item in result.allocations)


def test_risk_based_without_stop_loss_returns_zero_for_that_asset() -> None:
    result = size_positions(
        _request(stops=(None, 45.0)),
        PositionSizingConfig(method="risk_based"),
    )

    positions = {item.asset_symbol: item for item in result.allocations}
    assert result.status == "PARTIAL"
    assert positions["AAA"].status == "ZERO_SIZE"
    assert positions["AAA"].target_quantity == 0.0
    assert positions["BBB"].target_quantity > 0.0


def test_liquidity_cap_and_lot_rounding_are_applied_without_rounding_up() -> None:
    result = size_positions(
        _request(volumes=(100.0, 100.0)),
        PositionSizingConfig(method="fixed_fractional", max_participation_rate=0.05),
    )

    for item in result.allocations:
        assert item.target_quantity <= 5.0
        assert item.target_quantity == pytest.approx(round(item.target_quantity))
        assert item.adv_participation <= 0.05 + 1e-12
        assert item.liquidity_limited


def test_confidence_weighting_scales_the_upstream_target() -> None:
    result = size_positions(
        _request(),
        PositionSizingConfig(method="confidence_weighting", maximum_position_weight=1.0),
    )
    positions = {item.asset_symbol: item for item in result.allocations}
    source_weights = {
        item.asset_symbol: item.target_weight
        for item in _request().portfolio_optimization.allocations
    }
    assert positions["AAA"].target_weight == pytest.approx(source_weights["AAA"])
    assert positions["BBB"].target_weight == pytest.approx(source_weights["BBB"] * 0.5)


def test_nonpositive_expected_excess_return_produces_zero_kelly_size() -> None:
    original = _request()
    assets = (
        original.assets[0].model_copy(update={"expected_return": -0.01}),
        original.assets[1],
    )
    request = original.model_copy(update={"assets": assets})
    result = size_positions(
        request,
        PositionSizingConfig(method="kelly", maximum_position_weight=1.0),
    )

    positions = {item.asset_symbol: item for item in result.allocations}
    assert positions["AAA"].target_quantity == 0.0
    assert positions["BBB"].target_quantity > 0.0
    assert result.status == "PARTIAL"


def test_future_market_snapshot_is_rejected() -> None:
    with pytest.raises(ValidationError, match="market_data_as_of cannot be later"):
        _request(market_data_as_of=AS_OF + timedelta(seconds=1))


def test_sizing_fingerprint_and_allocations_are_reproducible() -> None:
    request = _request()
    config = PositionSizingConfig(method="risk_based")
    first = size_positions(request, config)
    second = size_positions(request, config)

    assert first.sizing_fingerprint == second.sizing_fingerprint
    assert first.allocations == second.allocations
    assert first.total_target_notional == pytest.approx(second.total_target_notional)


def test_invalid_upstream_status_is_rejected() -> None:
    request = _request()
    failed = request.portfolio_optimization.model_copy(update={"status": "FAILED"})
    with pytest.raises(ValidationError, match="OPTIMAL or FEASIBLE"):
        request.model_copy(update={"portfolio_optimization": failed})


def test_config_rejects_non_semantic_version() -> None:
    with pytest.raises(ValidationError, match="semantic versioning"):
        PositionSizingConfig(version="latest")


def test_position_sizing_never_reinvests_residual_cash_to_normalize_weights() -> None:
    result = size_positions(
        _request(),
        PositionSizingConfig(method="fixed_fractional", fixed_fractional_weight=0.03),
    )

    assert result.total_target_weight < 1.0
    assert all(item.target_weight <= 0.03 + 1e-8 for item in result.allocations)


def test_type_alias_is_explicitly_supported() -> None:
    request = _request()
    result = size_positions(request, PositionSizingConfig(method="fractional_kelly"))
    assert result.config_id == "position-sizer"
    assert result.base_currency == "COP"
