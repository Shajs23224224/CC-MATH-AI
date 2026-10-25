from datetime import UTC, datetime
from pathlib import Path
import subprocess

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


def _mock_cli_csv(monkeypatch, csv_text: str, *, returncode: int = 0, stderr: str = "") -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = Path(command[command.index("--directory") + 1])
        (output_dir / "candles.csv").write_text(csv_text, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command, returncode=returncode, stdout="", stderr=stderr
        )

    monkeypatch.setattr("data.providers.dukascopy.subprocess.run", fake_run)


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

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("CLI must not run when the exact request is cached")

    monkeypatch.setattr("data.providers.dukascopy.subprocess.run", fail_if_called)
    frame = provider.fetch(_request())
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(frame) == 1
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2024-01-02T00:00:00Z")


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Too Many Requests (HTTP 429)",
        "Headless browser could not acquire a WAF token (last HTTP status: 503)",
    ],
)
def test_provider_reports_rate_limit_or_waf_without_caching(
    tmp_path: Path, monkeypatch, diagnostic: str
) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    calls = []

    def fail_download(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            args=command, returncode=1, stdout="", stderr=diagnostic
        )

    monkeypatch.setattr("data.providers.dukascopy.subprocess.run", fail_download)

    with pytest.raises(DataProviderError, match="rate-limited or blocked"):
        provider.fetch(_request())

    assert len(calls) == 1
    assert not tmp_path.exists()


def test_provider_caches_valid_csv_only(tmp_path: Path, monkeypatch) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    _mock_cli_csv(
        monkeypatch,
        "timestamp,open,high,low,close,volume\n"
        "1704153600000,1.10,1.20,1.00,1.15,42\n",
    )

    frame = provider.fetch(_request())

    assert len(frame) == 1
    assert frame.iloc[0]["close"] == pytest.approx(1.15)
    assert list(tmp_path.glob("*.csv"))


def test_provider_does_not_cache_empty_csv(tmp_path: Path, monkeypatch) -> None:
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path)
    _mock_cli_csv(monkeypatch, "timestamp,open,high,low,close,volume\n")

    with pytest.raises(DataProviderError, match="no usable bars"):
        provider.fetch(_request())

    assert not tmp_path.exists()
