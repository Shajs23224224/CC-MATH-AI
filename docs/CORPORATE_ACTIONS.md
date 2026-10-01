# F10 — Corporate Actions

## Objective

Represent corporate events explicitly and provide deterministic split/reverse-split adjustment of historical OHLCV data.

## Supported event types

The domain model supports:

- split;
- reverse split;
- cash dividend;
- special dividend;
- spinoff;
- ticker change;
- merger.

Every event carries explicit provenance and relevant dates where available.

## Dates

The model separates:

- `announced_at`: when the event became known;
- `ex_date`: effective market date;
- `record_date`: holder eligibility date;
- `payable_date`: payment date.

This separation is required for point-in-time backtesting.

## Split mechanics

For a split with:

```text
ratio = new shares / old shares
```

historical bars strictly before the ex-date are transformed by:

```text
adjusted_price = raw_price / ratio
adjusted_volume = raw_volume * ratio
```

Examples:

- 2-for-1 split → prior prices divided by 2;
- 1-for-10 reverse split → prior prices multiplied by 10.

Bars on the ex-date are not adjusted by this routine because they are assumed to already reflect the market's ex-date trading convention.

## Dividends

F10 records cash dividends and special dividends as corporate events, but the split-adjustment routine deliberately does not subtract dividends from OHLC prices.

Dividend treatment will be implemented as an explicit cashflow/total-return methodology in later portfolio/return modules. This avoids silently mixing price-return and total-return semantics.

## Ticker changes, spinoffs and mergers

These events are preserved as structured events. Their full security-mapping, share-entitlement and accounting treatment requires later corporate-action-aware portfolio/accounting logic.

## Point-in-time eligibility

When `as_of` is supplied, events announced after `as_of` are not eligible.

This prevents future corporate announcements from entering historical decisions.

## Provider

Development provider:

```text
LocalCSVCoporateActionProvider
```

Expected file:

```text
data/local/corporate_actions/<SYMBOL>.csv
```

Minimum columns:

```text
action_type
source
```

Relevant optional columns:

```text
announced_at
ex_date
record_date
payable_date
ratio_numerator
ratio_denominator
cash_amount
currency
new_symbol
related_symbol
source_event_id
```

## F10 completion

- [x] Corporate-action type model
- [x] Point-in-time date fields
- [x] Corporate-action request/batch contracts
- [x] Provider protocol
- [x] Local CSV provider
- [x] Action normalization
- [x] Split and reverse-split adjustment
- [x] Dividend event preservation
- [x] Ticker/spinoff/merger event preservation
- [x] Unit tests
- [x] Integration test
- [x] Explicit adjustment semantics

**Status: COMPLETE**

Next phase: **F11 — Data Normalization**.
