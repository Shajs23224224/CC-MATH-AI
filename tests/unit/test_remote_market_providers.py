from __future__ import annotations

import pandas as pd
import pytest

from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from core.errors import DataProviderError
from data.providers.remote_market import (
    AlphaVantageMarketDataProvider,
    _window,
)


def _request() -> MarketDataRequest:
    return MarketDataRequest(
        asset=Asset(symbol="AAPL", asset_type=AssetType.EQUITY, currency="USD"),
        frequency=Frequency.DAILY,
        start=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
        end=pd.Timestamp("2024-01-03", tz="UTC").to_pydatetime(),
    )


def test_remote_frame_is_normalized_and_windowed() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [9.0, 10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0, 11.0],
            "close": [9.5, 10.5, 11.5, 12.5],
            "volume": [100.0, 110.0, 120.0, 130.0],
        }
    )
    result = _window(frame, _request())
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(result) == 2
    assert result["timestamp"].is_monotonic_increasing


def test_remote_frame_rejects_inconsistent_ohlc() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-02"],
            "open": [10.0],
            "high": [9.0],
            "low": [8.0],
            "close": [10.5],
            "volume": [100.0],
        }
    )
    with pytest.raises(DataProviderError, match="inconsistent OHLC high"):
        _window(frame, _request())


def test_alpha_vantage_requires_api_key_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    provider = AlphaVantageMarketDataProvider(api_key=None)
    with pytest.raises(DataProviderError, match="ALPHAVANTAGE_API_KEY"):
        provider.fetch(_request())
