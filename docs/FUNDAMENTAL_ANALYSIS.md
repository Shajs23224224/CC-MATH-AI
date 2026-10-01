# F17 — Fundamental Analysis

F17 adds deterministic fundamental-analysis metrics on top of the point-in-time FundamentalSnapshot contract.

## Implemented metrics

- ROE = net income / equity.
- ROA = net income / total assets.
- ROIC = NOPAT / invested capital, with an explicit tax-rate input.
- ROCE = EBIT / (total assets - current liabilities).
- EBITDA, EBIT, net-income and FCF margins.
- Cash conversion = FCF / net income.
- Payout = dividends / net income.
- Retention = 1 - payout.
- Debt-to-equity and debt-to-assets leverage.
- Interest coverage = EBIT / interest expense.
- Growth for revenue, EBITDA, EPS and FCF.

All ratio functions return decimal fractions rather than percentage points.

## Point-in-time safety

fundamental_growth requires an explicit as_of date.

For every accounting period it:
1. excludes snapshots reported after as_of;
2. keeps the latest report available by as_of for that period;
3. orders observations by period_end;
4. computes growth only from the selected point-in-time values.

This prevents a later restatement from leaking into an earlier historical view.

## Data requirements

F17 extends FundamentalSnapshot with:
- total_assets;
- current_liabilities;
- interest_expense.

These are optional at the contract boundary but are required by the specific metrics that need them. Missing values are rejected by the metric function; no silent imputation is performed.

## Numerical and domain rules

- Ratio denominators cannot be zero.
- ROA, ROIC and ROCE require a positive denominator.
- Growth cannot use a zero prior value.
- Non-finite numeric inputs are rejected.
- Snapshots from different assets cannot be combined.
- Duplicate reports for the same period/report date are rejected by the growth engine.

F17 remains a mathematical/analytics layer. It does not generate trading decisions, portfolio weights or execution orders.
