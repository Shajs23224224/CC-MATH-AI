import subprocess
from pathlib import Path

from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from data import MarketDataEngine
from data.providers import DukascopyMarketDataProvider, LocalCSVMarketDataProvider


def test_market_data_pipeline_preserves_request_frequency(tmp_path) -> None:
    path = tmp_path / "MSFT.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-02T00:00:00Z,100,102,99,101,500\n"
        "2026-01-05T00:00:00Z,101,104,100,103,800\n",
        encoding="utf-8",
    )

    request = MarketDataRequest(
        asset=Asset(symbol="MSFT", asset_type=AssetType.EQUITY, currency="USD"),
        frequency=Frequency.DAILY,
    )
    bars, batch = MarketDataEngine(LocalCSVMarketDataProvider(tmp_path)).fetch(request)

    assert request.frequency == Frequency.DAILY
    assert batch.bars_count == len(bars) == 2


def test_dukascopy_provider_integrates_with_market_data_engine(
    tmp_path, monkeypatch
) -> None:
    request = MarketDataRequest(
        asset=Asset(symbol="EURUSD", asset_type=AssetType.FX, currency="USD"),
        frequency=Frequency.MINUTE_15,
        start="2024-01-02T00:00:00Z",
        end="2024-01-04T00:00:00Z",
    )

    def fake_cli(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = Path(command[command.index("--directory") + 1])
        (output_dir / "candles.csv").write_text(
            "timestamp,open,high,low,close,volume\\n"
            "1704153600000,1.10,1.20,1.00,1.15,42\\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr("data.providers.dukascopy.subprocess.run", fake_cli)
    provider = DukascopyMarketDataProvider(cache_dir=tmp_path / "cache")
    bars, batch = MarketDataEngine(provider).fetch(request)

    assert batch.bars_count == len(bars) == 1
    assert bars[0].asset.symbol == "EURUSD"
    assert bars[0].timestamp.isoformat() == "2024-01-02T00:00:00+00:00"
    assert bars[0].close == 1.15
    assert list((tmp_path / "cache").glob("*.csv"))

