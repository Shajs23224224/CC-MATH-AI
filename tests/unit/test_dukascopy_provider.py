from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess

import pandas as pd
import pytest

from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from core.errors import DataProviderError
from data.providers.dukascopy import DukascopyMarketDataProvider


def _request(**overrides: object) -> MarketDataRequest:
    values: dict[str, object] = {
        "asset": Asset(symbol="EURUSD", asset_type=AssetType.FX, currency="USD"),
        "frequency": Frequency.MINUTE_15,
        "start": datetime(2024, 1, 2, tzinfo=UTC),
        "end": datetime(2024, 1, 4, tzinfo=UTC),
    }
    values.update(overrides)
    return MarketDataRequest(**values)


def test_provider_maps_frequency_and_exposes_supported_symbols(tmp_path: Path) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    assert Frequency.MINUTE_15 in provider.supported_frequencies
    assert Frequency.DAILY in provider.supported_frequencies
    assert Frequency.WEEKLY not in provider.supported_frequencies


def test_provider_rejects_unbounded_downloads(tmp_path: Path) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    with pytest.raises(DataProviderError, match="both start and end"):
        provider.fetch(_request(start=None))


def test_provider_rejects_unsupported_symbol(tmp_path: Path) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    request = _request(asset=Asset(symbol="UNKNOWN", asset_type=AssetType.FX, currency="USD"))
    with pytest.raises(DataProviderError, match="unsupported Dukascopy instrument"):
        provider.fetch(request)


def test_provider_rejects_adjusted_prices(tmp_path: Path) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    with pytest.raises(DataProviderError, match="unadjusted"):
        provider.fetch(_request(adjusted=True))


def test_provider_reads_cached_data_without_running_cli(tmp_path: Path, monkeypatch) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    cache = tmp_path / "eurusd_m15_20240102T000000_20240104T000000.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-02T00:00:00Z"],
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
            "volume": [42.0],
        }
    ).to_csv(cache, index=False)

    def fail_if_called(*args: object, **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("CLI must not run when the exact request is cached")

    monkeypatch.setattr("data.providers.dukascopy.subprocess.run", fail_if_called)
    frame = provider.fetch(_request())
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(frame) == 1
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2024-01-02T00:00:00Z")
