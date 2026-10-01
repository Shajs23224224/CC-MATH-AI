from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from data import MarketDataEngine
from data.providers import LocalCSVMarketDataProvider


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
