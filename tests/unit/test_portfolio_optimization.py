"""Behavioral and numerical tests for F36 portfolio optimization."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from core.contracts import (
    DecisionRecord,
    DependenceMatrix,
    SignalType,
)
from portfolio.optimization import (
    OptimizationMethod,
    PortfolioAssetInput,
    PortfolioOptimizationConfig,
    PortfolioOptimizationRequest,
    optimize_portfolio,
)
from signals.decision import DecisionOutput, DecisionProbabilities

AS_OF = datetime(2026, 10, 23, 15, 0, tzinfo=UTC)
SYMBOLS = ("AAA", "BBB", "CCC")
EXPECTED_RETURNS = (0.12, 0.08, 0.04)
COVARIANCE_VALUES = (
    (0.0400, 0.0060, 0.0020),
    (0.0060, 0.0225, 0.0030),
    (0.0020, 0.0030, 0.0100),
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decision(
    symbol: str,
    expected_return: float,
    *,
    signal: SignalType = SignalType.BUY,
    timestamp: datetime = AS_OF - timedelta(minutes=10),
    abstained: bool = False,
    horizon: str = "1D",
) -> DecisionOutput:
    """Create a small, schema-valid F35 decision fixture."""
    probability_map = {
        SignalType.BUY: DecisionProbabilities(buy=0.8, hold=0.1, sell=0.1),
        SignalType.HOLD: DecisionProbabilities(buy=0.1, hold=0.8, sell=0.1),
        SignalType.SELL: DecisionProbabilities(buy=0.1, hold=0.1, sell=0.8),
    }
    probabilities = probability_map[signal]
    chosen_probability = probabilities.for_signal(signal)
    record = DecisionRecord(
        decision_id=f"decision-{symbol}",
        asset_symbol=symbol,
        timestamp=timestamp,
        signal=signal,
        signal_probability=chosen_probability,
        model_confidence=0.0 if abstained else 0.75,
        expected_return=expected_return,
        expected_risk=0.1,
        risk_adjusted_score=expected_return / 0.1,
        recommended_horizon=horizon,
        suggested_position_size=0.0,
        regime="SYNTHETIC",
        data_snapshot_id="synthetic-f35-snapshot",
        model_versions=("synthetic-model@1.0.0",),
        feature_versions=("synthetic-features@1.0.0",),
    )
    return DecisionOutput(
        record=record,
        probabilities=probabilities,
        status="ABSTAINED" if abstained else "DECIDED",
        abstained=abstained,
        abstention_reason="synthetic abstention fixture" if abstained else None,
        agreement=0.9,
        uncertainty=0.1,
        evidence=(),
        constraints=(),
        decision_config_id="test-decision",
        decision_config_version="1.0.0",
        probability_calibration_status="uncalibrated",
        ensemble_fingerprint=_sha(f"ensemble-{symbol}"),
        decision_fingerprint=_sha(f"decision-{symbol}"),
    )


def _covariance() -> DependenceMatrix:
    return DependenceMatrix(
        assets=SYMBOLS,
        method="sample_covariance",
        kind="covariance",
        observations=60,
        start_index=0,
        end_index=60,
        matrix=COVARIANCE_VALUES,
    )


def _request(
    *,
    current_weights: tuple[float, ...] = (0.0, 0.0, 0.0),
    signals: tuple[SignalType, ...] = (
        SignalType.BUY,
        SignalType.BUY,
        SignalType.BUY,
    ),
    expected_returns: tuple[float, ...] = EXPECTED_RETURNS,
    sectors: tuple[str | None, ...] = ("TECH", "TECH", "HEALTH"),
    minima: tuple[float, ...] = (0.0, 0.0, 0.0),
    maxima: tuple[float, ...] = (0.8, 0.8, 0.8),
    eligible: tuple[bool, ...] = (True, True, True),
    abstained: tuple[bool, ...] = (False, False, False),
    as_of: datetime = AS_OF,
    covariance_as_of: datetime = AS_OF - timedelta(hours=1),
    horizon: str = "1D",
    view_matrix: tuple[tuple[float, ...], ...] = (),
    view_returns: tuple[float, ...] = (),
    view_confidences: tuple[float, ...] = (),
    equilibrium_weights: tuple[float, ...] = (),
) -> PortfolioOptimizationRequest:
    assets = tuple(
        PortfolioAssetInput(
            decision=_decision(
                symbol,
                expected_return,
                signal=signal,
                abstained=is_abstained,
                horizon=horizon,
            ),
            current_weight=current_weight,
            minimum_weight=minimum,
            maximum_weight=maximum,
            sector=sector,
            eligible=is_eligible,
        )
        for (
            symbol,
            expected_return,
            signal,
            current_weight,
            minimum,
            maximum,
            sector,
            is_eligible,
            is_abstained,
        ) in zip(
            SYMBOLS,
            expected_returns,
            signals,
            current_weights,
            minima,
            maxima,
            sectors,
            eligible,
            abstained,
            strict=True,
        )
    )
    return PortfolioOptimizationRequest(
        assets=assets,
        covariance=_covariance(),
        as_of=as_of,
        horizon=horizon,
        covariance_as_of=covariance_as_of,
        covariance_horizon="1D",
        data_snapshot_id="decision-snapshot-1",
        covariance_data_snapshot_id="covariance-snapshot-1",
        view_matrix=view_matrix,
        view_returns=view_returns,
        view_confidences=view_confidences,
        equilibrium_weights=equilibrium_weights,
    )


def _config(
    method: OptimizationMethod = "mean_variance",
    **overrides: Any,
) -> PortfolioOptimizationConfig:
    return PortfolioOptimizationConfig(method=method, **overrides)


@pytest.mark.parametrize(
    "method",
    (
        "equal_weight",
        "mean_variance",
        "minimum_variance",
        "maximum_sharpe",
        "risk_parity",
        "inverse_volatility",
        "hrp",
        "black_litterman",
        "maximum_diversification",
    ),
)
def test_all_nine_methods_return_a_valid_fully_invested_allocation(
    method: OptimizationMethod,
) -> None:
    if method == "black_litterman":
        request = _request(
            view_matrix=((1.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            view_returns=(0.02, 0.04),
            view_confidences=(0.8, 0.7),
            equilibrium_weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        )
    else:
        request = _request()

    result = optimize_portfolio(request, _config(method))

    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert sum(item.target_weight for item in result.allocations) == pytest.approx(1.0)
    assert all(0.0 <= item.target_weight <= 1.0 for item in result.allocations)
    assert all(item.satisfied for item in result.constraints)
    assert result.expected_return is not None
    assert result.volatility is not None and result.volatility > 0.0
    assert len(result.optimization_fingerprint) == 64
    assert result.method == method


def test_equal_weight_method_matches_equal_weights_without_binding_limits() -> None:
    request = _request(maxima=(1.0, 1.0, 1.0))

    result = optimize_portfolio(request, _config("equal_weight"))

    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert [item.target_weight for item in result.allocations] == pytest.approx(
        [1.0 / 3.0] * 3, abs=1e-5
    )


def test_sell_and_abstained_decisions_are_excluded() -> None:
    request = _request(
        current_weights=(0.5, 0.3, 0.2),
        signals=(SignalType.BUY, SignalType.SELL, SignalType.BUY),
        maxima=(1.0, 0.8, 0.8),
        abstained=(False, False, True),
    )

    result = optimize_portfolio(request, _config("minimum_variance"))

    assert result.status in {"OPTIMAL", "FEASIBLE"}
    weights = {item.asset_symbol: item.target_weight for item in result.allocations}
    assert weights["BBB"] == 0.0
    assert weights["CCC"] == 0.0
    assert {item.asset_symbol for item in result.exclusions} == {"BBB", "CCC"}


def test_hold_can_retain_but_not_increase_an_existing_position() -> None:
    request = _request(
        current_weights=(0.4, 0.4, 0.2),
        signals=(SignalType.BUY, SignalType.BUY, SignalType.HOLD),
    )

    result = optimize_portfolio(request, _config("mean_variance"))

    weights = {item.asset_symbol: item.target_weight for item in result.allocations}
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert weights["CCC"] <= 0.2 + 1e-6


def test_sector_cap_and_maximum_turnover_are_enforced() -> None:
    request = _request(current_weights=(0.25, 0.20, 0.55))
    config = _config(
        "mean_variance",
        sector_limits=({"sector": "TECH", "maximum_weight": 0.45},),
        max_turnover=0.2,
        max_concentration_hhi=0.55,
        transaction_cost_bps=8.0,
    )

    result = optimize_portfolio(request, config)

    assert result.status in {"OPTIMAL", "FEASIBLE"}
    weights = {item.asset_symbol: item.target_weight for item in result.allocations}
    assert weights["AAA"] + weights["BBB"] <= 0.45 + 1e-5
    assert result.turnover is not None and result.turnover <= 0.2 + 1e-5
    assert result.concentration_hhi is not None and result.concentration_hhi <= 0.55 + 1e-5
    assert result.estimated_transaction_cost_fraction is not None
    assert result.estimated_transaction_cost_fraction >= 0.0


def test_bounds_that_cannot_sum_to_one_return_infeasible_without_fake_weights() -> None:
    request = _request(minima=(0.4, 0.4, 0.4), maxima=(0.8, 0.8, 0.8))

    result = optimize_portfolio(request, _config("minimum_variance"))

    assert result.status == "INFEASIBLE"
    assert all(item.target_weight == 0.0 for item in result.allocations)
    assert result.expected_return is None
    assert "cannot sum to a fully invested portfolio" in result.solver_message


def test_positive_minimum_weight_for_sell_signal_is_infeasible() -> None:
    request = _request(
        minima=(0.0, 0.2, 0.0),
        signals=(SignalType.BUY, SignalType.SELL, SignalType.BUY),
    )

    result = optimize_portfolio(request, _config("equal_weight"))

    assert result.status == "INFEASIBLE"
    assert "positive minimum weight" in result.solver_message


def test_black_litterman_requires_explicit_views() -> None:
    result = optimize_portfolio(_request(), _config("black_litterman"))

    assert result.status == "FAILED"
    assert "requires at least one explicit view" in result.solver_message


def test_equal_weight_results_are_reproducible() -> None:
    request = _request(maxima=(1.0, 1.0, 1.0))
    config = _config("equal_weight")

    first = optimize_portfolio(request, config)
    second = optimize_portfolio(request, config)

    assert first.optimization_fingerprint == second.optimization_fingerprint
    assert first.allocations == second.allocations
    assert first.objective_value == pytest.approx(second.objective_value)


def test_cash_deployment_is_turnover_but_cash_is_not_charged_commission() -> None:
    request = _request()
    result = optimize_portfolio(
        request,
        _config("equal_weight", transaction_cost_bps=10.0),
    )

    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.turnover == pytest.approx(1.0, abs=1e-5)
    assert result.estimated_transaction_cost_fraction == pytest.approx(0.001, abs=1e-7)


def test_future_covariance_and_mismatched_horizons_are_rejected() -> None:
    with pytest.raises(ValidationError, match="covariance_as_of"):
        _request(covariance_as_of=AS_OF + timedelta(seconds=1))

    with pytest.raises(ValidationError, match="horizons must match"):
        _request(horizon="1W")
