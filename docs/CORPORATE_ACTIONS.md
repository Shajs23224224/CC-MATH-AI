# F10 — Corporate Actions

## Objective

Represent corporate events explicitly and provide deterministic split/reverse-split adjustment of historical OHLCV data.

## Supported event types

- split
- reverse split
- cash dividend
- special dividend
- spinoff
- ticker change
- merger

Every event carries provenance plus relevant announcement/effective dates.

## Date semantics

- `announced_at`: when the event became known;
- `ex_date`: market-effective date;
- `record_date`: eligibility date;
- `payable_date`: payment date.

For point-in-time requests with `as_of`, events must have an announcement date on or before `as_of`. Events without announcement provenance are excluded rather than treated as known.

## Split mechanics

For a split:

```text
ratio = new shares / old shares
adjusted_price = raw_price / ratio
adjusted_volume = raw_volume * ratio
```

Historical bars strictly before the ex-date are adjusted.

Examples:

- 2-for-1 split: prior prices / 2 and prior volume * 2;
- 1-for-10 reverse split: prior prices * 10 and prior volume / 10.

Bars on the ex-date are not modified by this routine.

## Dividends

Cash and special dividends are stored as explicit events. They are **not** silently subtracted from OHLC prices.

Price-return and total-return methodologies must remain distinct. Dividend cashflows will be incorporated explicitly in later return/portfolio accounting work.

## Other events

Spinoffs, mergers and ticker changes are represented as structured events but do not yet mutate positions or security identity automatically. Those mappings require portfolio/accounting semantics beyond simple OHLCV scaling.

## Provider

Development adapter:

```text
LocalCSVCorporateActionProvider
```

Input:

```text
data/local/corporate_actions/<SYMBOL>.csv
```

Minimum columns:

```text
action_type
source
```

Optional columns include:

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
- [x] Request and batch contracts
- [x] Provider protocol
- [x] Local CSV provider
- [x] Action normalization
- [x] Split adjustment
- [x] Reverse-split adjustment
- [x] Dividend event preservation
- [x] Ticker/spinoff/merger preservation
- [x] Point-in-time announcement filtering
- [x] Unit tests
- [x] Integration test

**Status: COMPLETE**

Next phase: **F11 — Data Normalization**.
