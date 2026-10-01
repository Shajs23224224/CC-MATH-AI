from datetime import date

import pytest

from core.contracts import Asset, AssetType, FundamentalSnapshot
from quant.fundamentals import (
    cash_conversion_ratio,
    debt_to_assets,
    debt_to_equity,
    ebit_margin,
    ebitda_margin,
    free_cash_flow_margin,
    fundamental_growth,
    interest_coverage,
    net_margin,
    payout_ratio,
    retention_ratio,
    roa,
    roce,
    roe,
    roic,
)


def _asset(symbol: str = "AAPL") -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.EQUITY, currency="USD")


def _snapshot(
    *,
    period_end: date,
    reported_at: date,
    revenue: float = 1000.0,
    ebitda: float = 250.0,
    ebit: float = 200.0,
    net_income: float = 150.0,
    free_cash_flow: float = 130.0,
    cash: float = 600.0,
    debt: float = 300.0,
    equity: float = 900.0,
    total_assets: float = 1500.0,
    current_liabilities: float = 300.0,
    interest_expense: float = 20.0,
    dividends: float = 30.0,
    asset: Asset | None = None,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        asset=asset or _asset(),
        period_end=period_end,
        reported_at=reported_at,
        revenue=revenue,
        ebitda=ebitda,
        ebit=ebit,
        net_income=net_income,
        free_cash_flow=free_cash_flow,
        cash=cash,
        debt=debt,
        equity=equity,
        total_assets=total_assets,
        current_liabilities=current_liabilities,
        interest_expense=interest_expense,
        dividends=dividends,
    )


def test_core_fundamental_ratios() -> None:
    snapshot = _snapshot(
        period_end=date(2026, 6, 30),
        reported_at=date(2026, 7, 30),
    )

    assert roe(snapshot) == pytest.approx(1 / 6)
    assert roa(snapshot) == pytest.approx(0.10)
    assert roic(snapshot, 0.25) == pytest.approx(0.25)
    assert roce(snapshot) == pytest.approx(1 / 6)
    assert ebitda_margin(snapshot) == pytest.approx(0.25)
    assert ebit_margin(snapshot) == pytest.approx(0.20)
    assert net_margin(snapshot) == pytest.approx(0.15)
    assert free_cash_flow_margin(snapshot) == pytest.approx(0.13)
    assert cash_conversion_ratio(snapshot) == pytest.approx(130 / 150)
    assert payout_ratio(snapshot) == pytest.approx(0.20)
    assert retention_ratio(snapshot) == pytest.approx(0.80)
    assert debt_to_equity(snapshot) == pytest.approx(1 / 3)
    assert debt_to_assets(snapshot) == pytest.approx(0.20)
    assert interest_coverage(snapshot) == pytest.approx(10.0)


def test_growth_is_point_in_time_safe() -> None:
    snapshots = (
        _snapshot(
            period_end=date(2026, 3, 31),
            reported_at=date(2026, 4, 25),
            revenue=1000.0,
        ),
        _snapshot(
            period_end=date(2026, 6, 30),
            reported_at=date(2026, 7, 28),
            revenue=1200.0,
        ),
        _snapshot(
            period_end=date(2026, 6, 30),
            reported_at=date(2026, 9, 15),
            revenue=1250.0,
        ),
        _snapshot(
            period_end=date(2026, 9, 30),
            reported_at=date(2026, 10, 25),
            revenue=1400.0,
        ),
    )

    before_restatement = fundamental_growth(snapshots, "revenue", as_of=date(2026, 8, 31))
    assert len(before_restatement) == 2
    assert before_restatement[1].value == 1200.0
    assert before_restatement[1].growth == pytest.approx(0.20)

    after_restatement = fundamental_growth(snapshots, "revenue", as_of=date(2026, 12, 31))
    assert len(after_restatement) == 3
    assert after_restatement[1].value == 1250.0
    assert after_restatement[1].reported_at == date(2026, 9, 15)
    assert after_restatement[1].growth == pytest.approx(0.25)


def test_growth_does_not_bridge_missing_values() -> None:
    snapshots = (
        _snapshot(
            period_end=date(2026, 3, 31),
            reported_at=date(2026, 4, 25),
            revenue=1000.0,
        ),
        _snapshot(
            period_end=date(2026, 6, 30),
            reported_at=date(2026, 7, 28),
            revenue=None,
        ),
        _snapshot(
            period_end=date(2026, 9, 30),
            reported_at=date(2026, 10, 25),
            revenue=1500.0,
        ),
    )
    points = fundamental_growth(snapshots, "revenue", as_of=date(2026, 12, 31))
    assert points[1].growth is None
    assert points[2].growth is None


def test_fundamental_invalid_denominators_are_explicit() -> None:
    snapshot = _snapshot(
        period_end=date(2026, 6, 30),
        reported_at=date(2026, 7, 30),
        total_assets=None,
    )
    with pytest.raises(ValueError, match="total_assets"):
        roa(snapshot)

    zero_interest = _snapshot(
        period_end=date(2026, 6, 30),
        reported_at=date(2026, 7, 30),
        interest_expense=0.0,
    )
    with pytest.raises(ValueError, match="denominator"):
        interest_coverage(zero_interest)

    with pytest.raises(ValueError, match="previous value"):
        from quant.fundamentals import growth_rate

        growth_rate(10.0, 0.0)


def test_growth_rejects_mixed_assets_and_unknown_metrics() -> None:
    first = _snapshot(period_end=date(2026, 3, 31), reported_at=date(2026, 4, 25))
    second = _snapshot(
        period_end=date(2026, 6, 30),
        reported_at=date(2026, 7, 28),
        asset=_asset("MSFT"),
    )
    with pytest.raises(ValueError, match="same asset"):
        fundamental_growth((first, second), "revenue", as_of=date(2026, 8, 1))

    with pytest.raises(ValueError, match="unsupported growth metric"):
        fundamental_growth((first,), "net_income", as_of=date(2026, 8, 1))
