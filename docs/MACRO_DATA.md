# F09 — Macro Data Engine

## Objective

Provide a generic, point-in-time macroeconomic data pipeline.

## Macro domains

The engine is designed to represent series such as:

- inflation;
- unemployment;
- GDP;
- policy rates;
- government yields;
- yield-curve observations;
- credit spreads;
- commodity macro series;
- macroeconomic indexes.

The engine does not hard-code one country or provider. A series is identified by a provider-defined series ID plus metadata.

## Point-in-time semantics

Macro data uses:

- `period_end`: period represented by the observation;
- `released_at`: date the observation became available.

For historical analysis at date T:

```text
released_at <= T
```

must hold.

This protects backtests from using macro releases that had not yet occurred.

## Generic observation model

Each observation records:

- series ID;
- series name;
- geography;
- period end;
- release date;
- numeric value;
- unit;
- frequency;
- source;
- optional tenor;
- optional linked asset.

The optional tenor supports yield-curve style series such as 3M, 2Y, 5Y or 10Y without requiring a separate domain model yet.

## Request model

A macro request contains:

- series ID;
- name;
- geography;
- optional period bounds;
- optional `as_of`;
- frequency.

## Local provider

F09 provides:

```text
LocalCSVMacroProvider
```

Expected file:

```text
data/local/macro/<SERIES_ID>.csv
```

Required columns:

```text
series_id
name
geography
period_end
released_at
value
unit
frequency
source
```

`tenor` is optional.

## Pipeline

```text
MacroDataRequest
    ↓
MacroDataProvider
    ↓
Raw DataFrame
    ↓
Date / numeric normalization
    ↓
Duplicate + validity checks
    ↓
Point-in-time filter
    ↓
MacroObservation[]
    ↓
MacroDataBatch
```

## Quality boundaries

F09 rejects:

- invalid dates;
- invalid numeric values;
- missing required fields;
- duplicate point-in-time observations;
- publication dates before the represented period.

F12 will extend macro quality handling with outliers, revision lineage, stale-data detection and broader anomaly controls.

## F09 completion

- [x] Generic macro request contract
- [x] Macro observation contract
- [x] Macro batch contract
- [x] Provider protocol
- [x] Local CSV provider
- [x] Normalization
- [x] Point-in-time `as_of` filtering
- [x] Yield-curve tenor field
- [x] Unit tests
- [x] Integration test
- [x] Look-ahead boundary

**Status: COMPLETE**

Next phase: **F10 — Corporate Actions**.
