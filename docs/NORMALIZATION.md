# F11 — Data Normalization

## Objective

Create one canonical normalization layer shared across market, fundamental, macro and corporate-action data.

## Canonical rules

### Symbols / identifiers

- trim surrounding whitespace;
- uppercase canonical symbols and identifiers;
- reject empty identifiers.

### Timestamps

- parse timestamps explicitly;
- attach UTC to timezone-naive values;
- convert timezone-aware values to UTC;
- preserve the instant in time rather than the source display timezone.

Example:

```
2026-10-01 12:00 -05:00
        ↓
2026-10-01 17:00 UTC
```

### Dates

Period/reporting/event dates are represented as calendar dates after timestamp parsing.

### Currencies

Currency codes are standardized to uppercase three-letter ISO-style codes.

Examples:

```
usd → USD
eur → EUR
cop → COP
```

F11 does **not** convert monetary values between currencies.

A value of 100 EUR remains 100 EUR after normalization. FX conversion requires an explicit FX dataset, timestamp, source and conversion policy.

### Units

Common aliases are canonicalized:

```
% / pct / percentage → percent
usd → USD
local_currency → LOCAL_CURRENCY
```

Unit naming normalization does not change the underlying numeric value.

### Frequencies

Market-frequency aliases are mapped to the canonical `Frequency` enum.

Macro-frequency aliases are mapped to the canonical `MacroFrequency` enum.

### Columns

DataFrame column names are normalized by:

- trimming;
- lowercasing;
- replacing spaces/hyphens with underscores;
- applying domain aliases.

Collisions after normalization are rejected.

## Domain coverage

F11 is applied to:

- market OHLCV normalization;
- fundamental data normalization;
- macro observations;
- corporate actions and split-adjustment inputs.

## Provenance

Normalization must not erase source semantics needed for auditability.

Raw/source data remains the provider boundary; normalized records are the canonical domain representation.

## Conversion policy

F11 distinguishes:

```
Normalization
    ↓
same information, canonical representation

Conversion
    ↓
numeric transformation using an explicit rule
```

Currency conversion, inflation adjustment, unit scaling and corporate-action adjustment are therefore explicit operations, not hidden side effects of generic normalization.

## F11 completion

- [x] Canonical symbols/identifiers
- [x] UTC timestamp normalization
- [x] Calendar-date normalization
- [x] Currency-code normalization
- [x] Unit normalization
- [x] Market-frequency normalization
- [x] Macro-frequency normalization
- [x] Column normalization
- [x] Collision detection
- [x] Typed normalization report
- [x] Cross-domain integration
- [x] Unit tests

**Status: COMPLETE**

Next phase: **F12 — Data Quality + Feature Store**.
