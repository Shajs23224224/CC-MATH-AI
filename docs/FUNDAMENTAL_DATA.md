# F08 — Fundamental Data Engine

## Objective

Add a point-in-time fundamental-data pipeline for financial statement and per-share information while preserving reporting-date provenance.

## Fundamental fields

The canonical FundamentalSnapshot supports:
- revenue;
- EBITDA;
- EBIT;
- net income;
- EPS;
- free cash flow;
- CAPEX;
- cash;
- debt;
- equity;
- dividends;
- shares outstanding.

## Point-in-time semantics

Two dates are mandatory:
- period_end: end of the accounting period represented by the data;
- reported_at: date on which the information became available.

For an analysis date T, only observations satisfying reported_at <= T are eligible.

This is the fundamental-data anti-look-ahead boundary.

## Pipeline

FundamentalDataRequest -> FundamentalDataProvider -> raw data -> normalization -> quality checks -> FundamentalSnapshot[] -> FundamentalDataBatch

## Provider

F08 provides LocalCSVFundamentalProvider.

Expected file:
data/local/fundamentals/<SYMBOL>.csv

Minimum columns:
period_end
reported_at

All supported fundamental fields are optional after the two required dates, allowing partial datasets.

## Request controls

FundamentalDataRequest supports asset, accounting-period bounds, as_of availability date and result limit.

## Normalization rules

The normalizer parses dates, standardizes numeric fields, rejects invalid required dates, rejects duplicate (period_end, reported_at) snapshots, sorts by publication date and accounting period, and creates immutable FundamentalSnapshot contracts.

## Look-ahead protection

The provider filters rows by reported_at when as_of is supplied.

The engine performs a second validation that returned snapshots do not violate as_of. This redundancy is intentional so an adapter bug cannot silently create future-information leakage.

## Revisions

Multiple observations for the same accounting period are allowed when their reported_at dates differ. This allows future filing revisions or amended observations to be represented as separate point-in-time records.

## Testing

F08 includes tests for reporting-date validation, normalization, duplicate snapshot detection, point-in-time filtering, missing-provider-file handling, local CSV ingestion, and integration flow.

## F08 completion

- [x] Fundamental snapshot contract
- [x] Fundamental request contract
- [x] Fundamental batch contract
- [x] Provider protocol
- [x] Local CSV adapter
- [x] Numeric/date normalization
- [x] Point-in-time as_of filtering
- [x] Duplicate detection
- [x] Engine integration
- [x] Unit tests
- [x] Integration test
- [x] Explicit look-ahead boundary

**Status: COMPLETE**

Next phase: F09 — Macro Data Engine
