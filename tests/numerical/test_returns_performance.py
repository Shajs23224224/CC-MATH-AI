"""Reference and invariant tests for F14 returns and performance."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.contracts import Asset, AssetType, PerformanceMetrics, PriceSeries, ReturnSeries
from quant.returns import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    downside_deviation,
    drawdown_series,
    log_return,
    log_returns,
    max_drawdown,
    performance_metrics,
    sharpe_ratio,
    simple_return,
    simple_return_series,
    simple_returns,
    sortino_ratio,
    wealth_index,
)


def _price_series() -> PriceSeries:
    asset = Asset(
        symbol="TEST",
        asset_type=AssetType.EQUITY,
        currency="USD",
        exchange="X",
    )
    return PriceSeries(
        asset=asset,
        timestamps=(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
        ),
        values=(100.0, 110.0, 99.0),
        frequency="1D",
    )


def test_single_period_returns() -> None:
    assert simple_return(100.0, 110.0) == pytest.approx(0.10)
    assert log_return(100.0, 110.0) == pytest.approx(0.0953101798043249)


def test_return_series_from_prices() -> None:
    values = simple_returns((100.0, 110.0, 99.0))
    assert values == pytest.approx((0.10, -0.10))

    result = simple_return_series(_price_series())
    assert isinstance(result, ReturnSeries)
    assert result.values == pytest.approx((0.10, -0.10))
    assert result.timestamps == _price_series().timestamps[1:]


def test_log_returns_from_prices() -> None:
    result = log_returns((100.0, 110.0, 99.0))
    assert result == pytest.approx((0.0953101798043249, -0.1053605156578263))


def test_cumulative_and_wealth() -> None:
    returns = (0.10, -0.10)
    assert cumulative_return(returns) == pytest.approx(-0.01)
    assert wealth_index(returns) == pytest.approx((1.0, 1.1, 0.99))
    assert drawdown_series(returns) == pytest.approx((0.0, 0.0, -0.10))
    assert max_drawdown(returns) == pytest.approx(0.10)


def test_annualization() -> None:
    returns = (0.01,) * 12
    assert annualized_return(returns, 12.0) == pytest.approx(0.12682503013196977)


def test_risk_adjusted_metrics() -> None:
    returns = (0.02, -0.01, 0.03, 0.00)
    expected_volatility = pytest.approx(
        0.018257418583505537 * 4.0**0.5,
        rel=1e-12,
    )
    assert annualized_volatility(returns, 4.0) == expected_volatility
    assert sharpe_ratio(returns, 4.0) == pytest.approx(1.0954451150103324)
    assert downside_deviation(returns, 0.0) == pytest.approx(0.005)
    assert sortino_ratio(returns, 4.0) == pytest.approx(4.0)


def test_performance_summary() -> None:
    returns = (0.02, -0.01, 0.03, 0.00)
    metrics = performance_metrics(returns, 4.0)
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.observations == 4
    assert metrics.total_return == pytest.approx(0.040094)
    assert metrics.max_drawdown == pytest.approx(0.01)
    assert metrics.positive_period_ratio == pytest.approx(0.5)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: simple_return(0.0, 10.0),
        lambda: log_return(10.0, 0.0),
        lambda: simple_returns((100.0,)),
        lambda: log_returns((100.0,)),
        lambda: cumulative_return((-1.0,)),
        lambda: wealth_index((0.01,), 0.0),
        lambda: annualized_return((0.01,), 0.0),
        lambda: annualized_volatility((0.01,), 12.0),
        lambda: sharpe_ratio((0.01,), 12.0),
        lambda: sortino_ratio((0.01, 0.02), 12.0),
        lambda: simple_returns((100.0, float("nan"))),
    ],
)
def test_invalid_domains_are_rejected(operation: object) -> None:
    with pytest.raises(ValueError):
        operation()  # type: ignore[operator]
