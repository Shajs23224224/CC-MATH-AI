# F37 — Position Sizing

## Purpose and boundary

F37 turns an accepted F36 PortfolioOptimizationResult into target quantities and notionals using account equity, point-in-time market inputs, an explicit horizon, and a versioned sizing policy. It is a deterministic decision-support layer: it does not create or transmit orders, round a quantity upward through a limit, guarantee a fill, or replace F21 risk controls / the F38 Risk Governor.

Only an F36 result with status OPTIMAL or FEASIBLE, matching symbols and horizon, and no failed reported constraints can enter the sizing request. The optimizer's target weight is a hard upper bound for every asset. Sizing may reduce an allocation and leave cash unallocated; it never increases F36 weights or creates leverage.

## Methods

| Method | Sizing rule |
|---|---|
| fixed_fractional | Apply a fixed equity exposure fraction per asset, capped by F36 target weight and the configured maximum position weight. This is a notional-exposure convention, not a claim about stop-out risk. |
| volatility_based | Set desired weight to volatility budget / forecast standard deviation for the same horizon, then cap it by F36. |
| risk_based | Allocate a total equity risk budget pro rata to F36 target weights and divide each amount by the price-to-stop distance. Requires a stop below current price; absent stops produce zero size. |
| kelly | Use non-negative (expected_return - risk_free_rate) / volatility², capped by F36 and maximum position weight. |
| fractional_kelly | Apply the configured fractional-Kelly multiplier to the full-Kelly estimate before the same caps. |
| confidence_weighting | Multiply F36 target weight by an explicitly supplied confidence in [0, 1]. |

Expected returns, risk-free rate and volatility must use the same declared horizon. Kelly sizing is a mathematical heuristic under restrictive assumptions, not a calibrated guarantee of optimal real-world growth.

## Hard bounds and audit

The contract requires positive account equity and prices, finite values, positive average daily volume, a positive lot size, timezone-aware timestamps, a three-letter base currency, and exact symbol ordering from F36. Market data and optimization timestamps cannot be later than sizing as_of. Long-only stop prices must be below the supplied price.

The engine applies the tightest of the F36 target weight, maximum position weight and method-specific desired weight; computes units; rounds down to the lot size; and caps units at configured participation in average daily volume. A rounded trade below minimum notional becomes zero. There is no hidden re-normalization after caps: residual exposure remains cash. Results include per-asset target weight, quantity, notional, estimated stop loss when available, risk budget, liquidity participation, limiting flags, rationale, totals, upstream optimization fingerprint, market snapshot ID and canonical SHA-256 sizing fingerprint.

Statuses are SIZED (all rows positive), PARTIAL (some rows zero) and ZERO_SIZE (all rows zero). A zero-sized line is a valid conservative outcome. Invalid upstream states and incompatible input contracts raise validation errors instead of emitting tradable quantities.

## Limits

Average daily volume is an input estimate, not executable liquidity; participation caps cannot guarantee fills or prevent market impact. The implementation assumes long-only cash equity exposure and does not model FX conversion, taxes, brokerage fees, market impact, fractional-lot broker rules, shorting, derivatives margin, or order execution. Upstream data sources remain responsible for the validity and point-in-time lineage of supplied prices, expected returns, volatility, confidence and stop levels.

Synthetic tests check deterministic engineering behavior only. They do not establish profitability, calibrated forecasts, or investment suitability. F38's final veto/reduction and downstream execution checks remain authoritative.

## Validation

The F37 workflow runs on Python 3.11 and 3.12 with Ruff lint/format, mypy, six-method behavior tests, bounds/liquidity/rounding and temporal tests, and regressions for F36, F35, F34, F33 and F21–F23. The general repository CI and Security checks must also pass on the pull-request head.
