# Historical market-data API adapters

This change adds optional historical OHLCV adapters behind the existing
`MarketDataProvider` interface. The normalized output remains compatible with
`MarketDataEngine` and canonical `MarketBar` contracts.

## Providers

| Adapter | Data scope | Credentials | Notes |
|---|---|---|---|
| `YahooFinanceMarketDataProvider` | Equities, ETFs, indices, FX and some crypto symbols | Usually no API key | Symbol coverage, intraday lookback and availability are controlled by Yahoo Finance; not an exchange-grade feed |
| `CCXTHistoricalMarketDataProvider` | Crypto exchange OHLCV supported by CCXT | Usually public endpoints; exchange-specific limits apply | Defaults to Binance; use exchange-native symbols such as `BTC/USDT` |
| `AlphaVantageMarketDataProvider` | Equities and selected market time series | `ALPHAVANTAGE_API_KEY` | API quotas apply; adjusted OHLCV is supported for daily/weekly data, not intraday |

The adapters use lazy imports. The core application remains importable without
installing optional providers.

## Installation

Install the optional market-data dependencies inside the project's virtual
environment:

```bash
python -m pip install -e ".[market-data,dev]"
```

Alpha Vantage uses the Python standard library and needs no additional package.
Set its API key only in the shell/session that runs the application:

```bash
export ALPHAVANTAGE_API_KEY="your_key_here"
```

Never put API keys in source code, commits, logs, or shared reports.

## Examples

### Yahoo Finance — historical daily equity prices

```python
from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from data.ingestion.market import MarketDataEngine
from data.providers import YahooFinanceMarketDataProvider

request = MarketDataRequest(
    asset=Asset(symbol="AAPL", asset_type=AssetType.EQUITY, currency="USD"),
    frequency=Frequency.DAILY,
)
bars, batch = MarketDataEngine(YahooFinanceMarketDataProvider()).fetch(request)
print(batch.bars_count, batch.first_timestamp, batch.last_timestamp)
```

### CCXT — historical crypto candles

```python
from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from data.ingestion.market import MarketDataEngine
from data.providers import CCXTHistoricalMarketDataProvider

request = MarketDataRequest(
    asset=Asset(symbol="BTC/USDT", asset_type=AssetType.CRYPTO, currency="USDT",
                exchange="binance"),
    frequency=Frequency.HOUR_1,
)
bars, batch = MarketDataEngine(
    CCXTHistoricalMarketDataProvider(exchange_id="binance")
).fetch(request)
print(batch.bars_count, batch.first_timestamp, batch.last_timestamp)
```

### Alpha Vantage — historical daily adjusted prices

```python
from core.contracts import Asset, AssetType, Frequency, MarketDataRequest
from data.ingestion.market import MarketDataEngine
from data.providers import AlphaVantageMarketDataProvider

request = MarketDataRequest(
    asset=Asset(symbol="IBM", asset_type=AssetType.EQUITY, currency="USD"),
    frequency=Frequency.DAILY,
    adjusted=True,
)
bars, batch = MarketDataEngine(AlphaVantageMarketDataProvider()).fetch(request)
print(batch.bars_count, batch.first_timestamp, batch.last_timestamp)
```

## Historical-data cautions

- These are download adapters, not a guarantee that the provider has every requested
  candle or symbol. Inspect empty results, date coverage, and rate-limit errors.
- Intraday history and API quotas vary by provider and account. Alpha Vantage may
  return a quota message rather than a time-series payload.
- Yahoo Finance and exchange APIs can revise or omit data. Preserve provider name,
  request parameters, retrieval time, and a local immutable snapshot for reproducible
  experiments.
- Corporate-action adjustment semantics differ between providers. Do not mix adjusted
  and unadjusted price series in one training dataset without explicit normalization.
- Keep all model validation chronological. Provider history is input data, not proof
  that a strategy will be profitable.
- No adapter places orders or connects to live execution.
