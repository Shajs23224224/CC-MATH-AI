# F07 — Market Data Engine

## Objective

Provide the first working market-data pipeline from a provider adapter to canonical OHLCV domain contracts.

## Supported market-data concepts

F07 establishes the architecture for:

- equities;
- ETFs;
- indices;
- FX;
- rates;
- commodities;
- crypto.

The domain also recognizes these frequencies:

- tick;
- 1m;
- 5m;
- 15m;
- 1h;
- 4h;
- 1D;
- 1W.

F07 implements OHLCV bars. Tick-specific microstructure data remains a later extension.

## Pipeline

```text
MarketDataRequest
      ↓
MarketDataProvider
      ↓
Raw DataFrame
      ↓
Column Standardization
      ↓
Timestamp / Numeric Normalization
      ↓
Basic Quality Validation
      ↓
MarketBar[]
      ↓
MarketDataBatch
```

## Provider boundary

Providers implement the `MarketDataProvider` protocol.

The first adapter is:

```text
LocalCSVMarketDataProvider
```

Expected file location:

```text
data/local/market/<SYMBOL>.csv
```

Required columns:

```text
timestamp
open
high
low
close
volume
```

Common timestamp aliases accepted:

```text
date
datetime
time
```

CSV column names are normalized case-insensitively.

The local provider is for development and testing. It is not a production market-data feed.

## Normalization rules

The normalizer:

- standardizes OHLCV column names;
- parses timestamps;
- normalizes timestamps to UTC;
- converts OHLCV fields to numeric values;
- sorts timestamps ascending;
- rejects duplicate timestamps;
- rejects invalid/missing timestamps;
- rejects missing/non-numeric OHLCV values;
- constructs immutable `MarketBar` contracts;
- validates OHLC relationships.

## Adjustment boundary

F07 does not implement corporate-action adjustment.

Therefore:

```text
adjusted = true
       ↓
LocalCSV provider rejects request
```

Corporate-action handling belongs to **F10 — Corporate Actions**.

This prevents the engine from silently mixing raw and adjusted price semantics.

## Request contract

`MarketDataRequest` contains:

- asset;
- frequency;
- optional start;
- optional end;
- adjusted flag.

The request window must satisfy:

```text
end > start
```

when both are supplied.

## Output contract

The engine returns:

```text
tuple[MarketBar, ...]
MarketDataBatch
```

The batch records:

- original request;
- number of bars;
- first timestamp;
- last timestamp.

## Error boundaries

F07 introduces:

- `DataProviderError`
- `DataQualityError`

Provider errors include unavailable files or unsupported requests.

Quality errors include duplicate timestamps, invalid timestamps and malformed OHLCV data.

F12 will expand this into the full data-quality subsystem with warnings, outlier detection, leakage detection and quality scoring.

## Tests

Implemented:

- request-window validation;
- timestamp ordering;
- UTC normalization;
- duplicate detection;
- OHLC validation;
- missing-provider-file handling;
- adjusted-price boundary;
- complete local CSV ingestion;
- integration pipeline verification.

## F07 completion

- [x] Market-data request contract
- [x] Frequency model
- [x] Provider protocol
- [x] Local CSV adapter
- [x] OHLCV normalization
- [x] UTC normalization
- [x] Basic quality validation
- [x] Market-data engine
- [x] Unit tests
- [x] Integration test
- [x] Explicit corporate-action boundary

**Status: COMPLETE**

Next phase: **F08 — Fundamental Data Engine**.
