from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from core.contracts import (
    Asset,
    AssetType,
    BacktestResult,
    DataQualityStatus,
    DecisionRecord,
    Forecast,
    FundamentalSnapshot,
    MarketBar,
    Portfolio,
    PriceSeries,
    Signal,
    SignalType,
)


def test_asset_is_normalized_and_immutable() -> None:
    asset = Asset(symbol=" aapl ", asset_type=AssetType.EQUITY, currency="usd")
    assert asset.symbol == "AAPL"
    assert asset.currency == "USD"
    with pytest.raises(ValidationError):
        asset.symbol = "MSFT"


def test_market_bar_validates_ohlc() -> None:
    bar = MarketBar(
        asset=Asset(symbol="AAPL", asset_type=AssetType.EQUITY, currency="USD"),
        timestamp=datetime.now(timezone.utc),
        open=100,
        high=105,
        low=95,
        close=102,
        volume=1000,
    )
    bar.validate_ohlc()


def test_price_series_requires_alignment_and_order() -> None:
    series = PriceSeries(
        asset=Asset(symbol="AAPL", asset_type=AssetType.EQUITY, currency="USD"),
        timestamps=(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        values=(100.0, 101.0),
        frequency="1D",
        data_quality=DataQualityStatus.VALID,
    )
    series.validate_alignment()


def test_signal_rejects_invalid_probability() -> None:
    with pytest.raises(ValidationError):
        Signal(
            asset_symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            signal=SignalType.BUY,
            probability=1.2,
            confidence=0.8,
            forecast=Forecast(
                horizon="20D",
                expected_return=0.05,
                expected_risk=0.12,
                uncertainty=0.04,
            ),
        )


def test_portfolio_market_values() -> None:
    from core.contracts import Position

    portfolio = Portfolio(
        portfolio_id="paper-1",
        timestamp=datetime.now(timezone.utc),
        base_currency="USD",
        cash=1000,
        positions=(
            Position(
                asset_symbol="AAPL",
                quantity=2,
                average_price=90,
                market_price=100,
            ),
        ),
    )
    assert portfolio.gross_market_value == 200
    assert portfolio.net_market_value == 200


def test_backtest_result_rejects_negative_drawdown() -> None:
    with pytest.raises(ValidationError):
        BacktestResult(
            strategy_id="demo",
            start_date="2026-01-01",
            end_date="2026-02-01",
            initial_capital=10000,
            final_equity=10500,
            total_return=0.05,
            max_drawdown=-0.1,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            turnover=0.2,
            trades=10,
            transaction_costs=20,
        )


def test_fundamental_snapshot_preserves_reporting_dates() -> None:
    snapshot = FundamentalSnapshot(
        asset=Asset(symbol="MSFT", asset_type=AssetType.EQUITY, currency="USD"),
        period_end=date(2026, 6, 30),
        reported_at=date(2026, 7, 25),
        revenue=1000,
        shares_outstanding=100,
    )
    assert snapshot.reported_at > snapshot.period_end


def test_decision_record_contains_version_provenance() -> None:
    record = DecisionRecord(
        decision_id="dec-1",
        asset_symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        signal=SignalType.HOLD,
        signal_probability=0.5,
        model_confidence=0.6,
        expected_return=0.01,
        expected_risk=0.08,
        risk_adjusted_score=0.125,
        recommended_horizon="20D",
        suggested_position_size=0.02,
        regime="SIDEWAYS",
        data_snapshot_id="data-1",
        model_versions=("model-v1",),
        feature_versions=("features-v1",),
    )
    assert record.model_versions == ("model-v1",)
