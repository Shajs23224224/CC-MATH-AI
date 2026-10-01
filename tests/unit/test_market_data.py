from datetime import datetime, timezone

import pandas as pd
import pytest

from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from core.errors import DataQualityError, DataProviderError
from data import MarketDataEngine, normalize_market_frame
from data.providers import LocalCSVMarketDataProvider


def _asset() -> Asset:
    return Asset(symbol="AAPL", asset_type=AssetType.EQUITY, currency="USD")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [
                "2026-01-02T00:00:00Z",
                "2026-01-05T00:00:00Z",
            ],
            "Open": [100, 102],
            "High": [103, 105],
            "Low": [99, 101],
            "Close": [102, 104],
            "Volume": [1000, 1200],
        }
    )


def test_request_rejects_inverted_window() -> None:
    with pytest.raises(ValueError):
        MarketDataRequest(
            asset=_asset(),
            start=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_normalize_market_frame_returns_utc_sorted_bars() -> None:
    frame = _frame().iloc[::-1].reset_index(drop=True)
    bars = normalize_market_frame(frame, _asset(), Frequency.DAILY)

    assert len(bars) == 2
    assert bars[0].timestamp < bars[1].timestamp
    assert bars[0].timestamp.tzinfo is not None


def test_normalization_rejects_duplicate_timestamps() -> None:
    frame = _frame()
    frame.loc[1, "Date"] = frame.loc[0, "Date"]
    with pytest.raises(DataQualityError):
        normalize_market_frame(frame, _asset(), Frequency.DAILY)


def test_normalization_rejects_invalid_ohlc() -> None:
    frame = _frame()
    frame.loc[0, "High"] = 90
    with pytest.raises((DataQualityError, ValueError)):
        normalize_market_frame(frame, _asset(), Frequency.DAILY)


def test_local_csv_provider_requires_file(tmp_path) -> None:
    provider = LocalCSVMarketDataProvider(tmp_path)
    request = MarketDataRequest(asset=_asset())
    with pytest.raises(DataProviderError):
        provider.fetch(request)


def test_engine_fetches_local_csv(tmp_path) -> None:
    path = tmp_path / "AAPL.csv"
    _frame().to_csv(path, index=False)

    engine = MarketDataEngine(LocalCSVMarketDataProvider(tmp_path))
    bars, batch = engine.fetch(MarketDataRequest(asset=_asset()))

    assert batch.bars_count == 2
    assert len(bars) == 2
    assert batch.first_timestamp == bars[0].timestamp
