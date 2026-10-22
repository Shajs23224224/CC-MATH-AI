"""Contract and safety tests for F22 tail-risk outputs and scenarios."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.contracts import StressScenario, TailRiskEstimate
from risk.tail import evaluate_stress_scenario, stress_test_portfolio


def test_tail_risk_contract_enforces_cvar_and_monte_carlo_metadata() -> None:
    with pytest.raises(ValidationError, match="cvar must be greater"):
        TailRiskEstimate(
            method="historical",
            var=0.10,
            cvar=0.05,
            confidence_level=0.95,
            horizon=1,
            observations=100,
            methodology="invalid contract fixture",
        )

    with pytest.raises(ValidationError, match="require simulations and random_seed"):
        TailRiskEstimate(
            method="monte_carlo_normal",
            var=0.10,
            cvar=0.12,
            confidence_level=0.95,
            horizon=1,
            observations=100,
            methodology="missing reproducibility metadata",
        )


def test_stress_scenario_normalizes_symbols_and_rejects_invalid_shocks() -> None:
    scenario = StressScenario(
        scenario_id="crash",
        version="1.0.0",
        description="Broad sell-off",
        shocks={" aapl ": -0.20},
    )
    assert scenario.shocks == {"AAPL": -0.20}

    with pytest.raises(ValidationError, match="greater than -1.0"):
        StressScenario(
            scenario_id="impossible",
            version="1",
            description="Invalid asset return",
            shocks={"AAPL": -1.0},
        )


def test_stress_test_calculates_signed_exposure_losses() -> None:
    scenario = StressScenario(
        scenario_id="equity-crash",
        version="2026.1",
        description="Equity sell-off",
        shocks={"AAPL": -0.10, "TSLA": -0.50},
    )

    result = evaluate_stress_scenario(
        {"AAPL": 0.60, "TSLA": 0.40},
        scenario,
    )

    assert result.portfolio_return == pytest.approx(-0.26)
    assert result.portfolio_loss == pytest.approx(0.26)
    assert result.covered_positions == 2
    assert result.scenario_version == "2026.1"


def test_stress_test_fails_closed_for_missing_position_shock() -> None:
    scenario = StressScenario(
        scenario_id="partial",
        version="1",
        description="Incomplete scenario",
        shocks={"AAPL": -0.10},
    )

    with pytest.raises(ValueError, match="missing shocks for held positions"):
        evaluate_stress_scenario({"AAPL": 0.5, "MSFT": 0.5}, scenario)


def test_stress_test_rejects_empty_and_duplicate_scenarios() -> None:
    scenario = StressScenario(
        scenario_id="crash",
        version="1",
        description="Equity sell-off",
        shocks={"AAPL": -0.10},
    )
    with pytest.raises(ValueError, match="at least one stress scenario"):
        stress_test_portfolio({"AAPL": 1.0}, ())
    with pytest.raises(ValueError, match="duplicate stress scenario"):
        stress_test_portfolio({"AAPL": 1.0}, (scenario, scenario))
