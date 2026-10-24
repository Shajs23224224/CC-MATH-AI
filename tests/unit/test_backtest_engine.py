from datetime import UTC, datetime, timedelta

import pytest

from backtest.engine import BacktestConfig, Bar, run_backtest


def bars(*closes: float) -> list[Bar]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        Bar(start + timedelta(days=index), close, close, close, close, 1000.0)
        for index, close in enumerate(closes)
    ]


def test_signal_executes_at_next_bar_open_and_accounts_costs() -> None:
    series = bars(100.0, 110.0, 120.0)
    result = run_backtest(
        series,
        {series[0].timestamp: 10, series[1].timestamp: 0},
        BacktestConfig(initial_capital=10_000.0, commission_bps=0, slippage_bps=0),
    )
    assert result.fills[0].timestamp == series[1].timestamp
    assert result.fills[0].reference_price == 110.0
    assert result.fills[0].units_delta == 10
    assert result.fills[1].timestamp == series[2].timestamp
    assert result.final_equity == pytest.approx(10_100.0)
    assert result.total_return == pytest.approx(0.01)


def test_commission_and_slippage_reduce_equity() -> None:
    series = bars(100.0, 100.0, 100.0)
    result = run_backtest(
        series,
        {series[0].timestamp: 10, series[1].timestamp: 0},
        BacktestConfig(initial_capital=10_000, commission_bps=10, slippage_bps=10),
    )
    assert result.commissions > 0
    assert result.slippage_costs > 0
    assert result.final_equity < 10_000


def test_final_bar_signal_is_not_executed_without_next_open() -> None:
    series = bars(100.0, 105.0)
    result = run_backtest(series, {series[-1].timestamp: 5})
    assert result.trades == 0
    assert result.final_equity == pytest.approx(100_000.0)


def test_future_price_change_does_not_change_prior_fill() -> None:
    first = bars(100.0, 101.0, 102.0)
    second = bars(100.0, 101.0, 500.0)
    first_result = run_backtest(first, {first[0].timestamp: 1})
    second_result = run_backtest(second, {second[0].timestamp: 1})
    assert first_result.fills[0].fill_price == second_result.fills[0].fill_price
    assert first_result.fills[0].timestamp == second_result.fills[0].timestamp


def test_rejects_unsorted_or_duplicate_timestamps() -> None:
    series = bars(100.0, 101.0, 102.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_backtest([series[0], series[2], series[1]], {})


def test_rejects_invalid_prices_and_non_integer_positions() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        Bar(datetime.now(UTC), float("nan"), 1, 1, 1)
    series = bars(100.0, 101.0)
    with pytest.raises(ValueError, match="integer units"):
        run_backtest(series, {series[0].timestamp: 1.5})  # type: ignore[dict-item]


def test_drawdown_is_bounded_and_costs_are_nonnegative() -> None:
    series = bars(100.0, 120.0, 90.0)
    result = run_backtest(
        series,
        {series[0].timestamp: 1},
        BacktestConfig(initial_capital=1_000, commission_bps=0, slippage_bps=0),
    )
    assert 0 <= result.max_drawdown <= 1
    assert result.commissions >= 0
    assert result.slippage_costs >= 0
