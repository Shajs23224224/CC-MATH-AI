# F18 — Valuation

F18 implements deterministic fundamental and relative valuation models with explicit units, currency and versioned assumptions.

## Relative valuation

Implemented metrics:

- P/E.
- P/S.
- P/B.
- EV/EBITDA.
- EV/FCF.
- Earnings yield.
- Comparable-company multiple aggregation using mean and median.
- Comparable-company implied valuation.

Enterprise-value methods use the explicit bridge:
Equity Value = Enterprise Value - Debt + Cash

Debt and cash are applied exactly once by the valuation function. SOTP also refuses debt/cash inputs when segment values are already equity values, preventing double counting.

## Intrinsic valuation

Implemented:

- Gordon growth.
- Multi-period DDM.
- DCF with explicit forecast cash flows and terminal growth.
- Residual income.
- NAV.
- SOTP.
- Reverse DCF terminal-growth inference.
- DCF sensitivity matrix.

## Contracts

ValuationAssumptions requires:

- assumptions version;
- currency;
- optional discount rate;
- optional terminal growth rate;
- optional tax rate.

ValuationResult records the method, assumptions version, currency, valuation basis, unit and valuation bridge values.

SensitivityMatrix records both axes and validates matrix shape.

## Numerical rules

- Inputs must be finite.
- Rates are bounded and terminal growth must remain below the discount/required return.
- Required denominators and per-share quantities must be positive where mathematically necessary.
- Debt/cash are non-negative.
- No silent imputation is performed.
- Reverse DCF uses deterministic bisection with explicit bounds, tolerance and iteration limit.
- Sensitivity is computed directly from the same deterministic DCF equation.

## Methodological boundaries

F18 computes valuation evidence only. It does not choose a BUY/HOLD/SELL action, does not optimize a portfolio and does not bypass the future Risk Governor.

Valuation outputs must be interpreted together with the assumptions, data timestamp, and method limitations.
