from math import isclose

import pytest

from core.contracts import SensitivityMatrix, ValuationAssumptions
from quant.valuation import (
    comparable_valuation_result,
    dcf_result,
    dcf_sensitivity,
    dcf_value,
    ddm_result,
    ddm_value,
    earnings_yield,
    ev_to_ebitda,
    ev_to_fcf,
    gordon_growth_value,
    implied_equity_value_from_multiple,
    implied_terminal_growth,
    nav_result,
    nav_value,
    pb_ratio,
    peer_multiple_mean,
    peer_multiple_median,
    pe_ratio,
    ps_ratio,
    residual_income_value,
    sotp_equity_value,
)


def _assumptions() -> ValuationAssumptions:
    return ValuationAssumptions(
        assumptions_version="f18-test-v1",
        currency="USD",
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        tax_rate=0.25,
    )


def test_relative_multiples() -> None:
    assert pe_ratio(20.0, 2.0) == pytest.approx(10.0)
    assert ps_ratio(20.0, 4.0) == pytest.approx(5.0)
    assert pb_ratio(20.0, 10.0) == pytest.approx(2.0)
    assert ev_to_ebitda(900.0, 300.0, 100.0, 100.0) == pytest.approx(11.0)
    assert ev_to_fcf(900.0, 300.0, 100.0, 100.0) == pytest.approx(11.0)
    assert earnings_yield(20.0, 2.0) == pytest.approx(0.10)
    assert implied_equity_value_from_multiple(
        10.0, 100.0, basis="enterprise", debt=200.0, cash=50.0
    ) == pytest.approx(850.0)


def test_gordon_ddm_and_residual_income() -> None:
    assumptions = _assumptions()
    gordon = gordon_growth_value(1.03, 0.10, 0.03)
    assert gordon == pytest.approx(14.714285714285714)

    ddm = ddm_result((1.0, 1.1), assumptions)
    assert ddm.per_share_value == pytest.approx(ddm.value)
    assert ddm.value > gordon

    residual = residual_income_value(10.0, (1.0, 1.1), 0.10, 0.03)
    assert residual == pytest.approx(25.194805194805194)


def test_dcf_and_equity_bridge() -> None:
    assumptions = _assumptions()
    enterprise = dcf_value((110.0, 121.0), 0.10, 0.03)
    assert enterprise == pytest.approx(1670.7792207792207)

    result = dcf_result(
        (110.0, 121.0),
        debt=300.0,
        cash=100.0,
        shares_outstanding=100.0,
        assumptions=assumptions,
    )
    assert result.enterprise_value == pytest.approx(enterprise)
    assert result.equity_value == pytest.approx(1470.7792207792207)
    assert result.per_share_value == pytest.approx(14.707792207792207)


def test_nav_sotp_and_comparables() -> None:
    assumptions = _assumptions()
    assert nav_value(100.0, 40.0) == pytest.approx(60.0)
    assert sotp_equity_value(
        (70.0, 50.0),
        segment_basis="enterprise",
        debt=20.0,
        cash=10.0,
    ) == pytest.approx(110.0)

    comparable = comparable_valuation_result(
        10.0,
        100.0,
        assumptions,
        basis="enterprise",
        debt=20.0,
        cash=10.0,
        shares_outstanding=10.0,
    )
    assert comparable.enterprise_value == pytest.approx(1000.0)
    assert comparable.equity_value == pytest.approx(990.0)
    assert comparable.per_share_value == pytest.approx(99.0)

    assert peer_multiple_mean((10.0, 12.0, 14.0)) == pytest.approx(12.0)
    assert peer_multiple_median((10.0, 12.0, 14.0, 16.0)) == pytest.approx(13.0)


def test_reverse_dcf_recovers_terminal_growth() -> None:
    target = dcf_value((110.0, 121.0), 0.10, 0.03)
    inferred = implied_terminal_growth(target, (110.0, 121.0), 0.10)
    assert isclose(inferred, 0.03, rel_tol=0.0, abs_tol=1e-8)


def test_dcf_sensitivity_is_typed() -> None:
    assumptions = _assumptions()
    sensitivity = dcf_sensitivity(
        (110.0, 121.0),
        (0.09, 0.10, 0.11),
        (0.02, 0.03, 0.04),
        assumptions,
    )
    assert isinstance(sensitivity, SensitivityMatrix)
    assert sensitivity.row_parameter == "discount_rate"
    assert sensitivity.column_parameter == "terminal_growth_rate"
    assert len(sensitivity.values) == 3
    assert len(sensitivity.values[0]) == 3


def test_valuation_boundaries_reject_invalid_double_counting_and_rates() -> None:
    assumptions = _assumptions()
    with pytest.raises(ValueError, match="positive"):
        pe_ratio(20.0, 0.0)
    with pytest.raises(ValueError, match="lower"):
        gordon_growth_value(1.0, 0.05, 0.05)
    with pytest.raises(ValueError, match="equity values"):
        sotp_equity_value(
            (50.0,),
            segment_basis="equity",
            debt=10.0,
            cash=5.0,
        )
    with pytest.raises(ValueError, match="positive"):
        nav_result(100.0, 40.0, 0.0, assumptions)
