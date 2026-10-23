# F38 — Portfolio Risk Governor

## Purpose

F38 is the final deterministic risk gate between the F37 position-sizing result and any future execution adapter. It returns **ALLOW**, **REDUCE**, or **BLOCK**, the measurements and reasons supporting the decision, and the exact lot-sized quantities left eligible after the gate. It never submits an order.

The existing F21 evaluator remains the constraint engine. F38 composes its fail-closed constraint behavior with F22 tail-risk evidence, F23 pairwise correlation data, F36 portfolio targets, and F37 position quantities. There is no path in this API to turn a prior BLOCK into an ALLOW.

## Controls

| Control | Source | Default policy |
|---|---|---|
| Maximum target position weight | F37 equity-relative target weights | Hard block |
| Gross leverage | F37 target notional / equity | Hard block |
| Sector weight | F38 sector context aggregated from F37 target weights | Hard block |
| Concentration HHI | F37 target notionals | Soft breach reduces |
| Factor exposure | F38 sector/factor context | Soft breach reduces |
| Absolute pairwise correlation | F23 aligned pairwise correlation values | Soft breach reduces |
| Drawdown and daily loss | Timestamped external risk metrics | Hard block |
| Tail loss / CVaR | Timestamped F22-compatible metric, fraction of notional | Hard block |
| Liquidation horizon | F21 liquidity metric using ADV and participation cap | Soft breach reduces |
| Turnover | F37 target versus current notional | Soft breach reduces |
| Data and model quality | Timestamped scores in [0, 1] | Hard block when below floor |
| Input freshness | F37, market, correlation/risk snapshots, and each external metric | Stale or future evidence blocks |

Defaults are configurable and versioned. They are policy parameters, not universal investment recommendations. Inputs must use documented units and consistent as-of timestamps. Missing any required constraint metric, including missing correlation pairs for active positions or missing factor loadings for active positions, blocks the decision.

## Decision semantics

- **ALLOW**: all required controls are present, fresh, unit-consistent, and within limits. F37 quantities remain unchanged.
- **REDUCE**: only soft controls are breached. The configured reduction factor is applied to each F37 quantity, rounded down to the F38 lot size. Cash remains unallocated.
- **BLOCK**: any hard limit is breached, a required measure is absent, evidence is stale/future, or risk evaluation fails. Every approved quantity and notional is zero. The result is not an order and downstream modules must preserve the veto.

If lot rounding after a requested reduction leaves every position at zero, the final decision becomes BLOCK. F38 verifies that each approved quantity is a lot-size multiple and cannot increase the approved F37 notional or equity-relative weight.

## Evidence and reproducibility

RiskGovernorRequest carries F37 sizing, asset/sector/factor/ADV context, pairwise correlations from F23, timestamped risk metrics, snapshot IDs, and an explicit timezone-aware evaluation timestamp. Results contain derived and supplied metrics, triggered constraint IDs, reasons, approved lot-sized quantities, the source F37 fingerprint, snapshot lineage, policy ID/version, and a canonical SHA-256 governor fingerprint.

Required externally supplied metric names and units:
- max_drawdown: fraction (positive drawdown magnitude in [0, 1]).
- daily_loss_fraction: fraction (positive loss magnitude in [0, 1]).
- max_cvar_fraction: fraction_of_notional (non-negative expected shortfall loss fraction).
- data_quality_score: score_0_to_1.
- model_quality_score: score_0_to_1.

Sector labels, factor loadings, pairwise correlations, and ADV are caller-supplied point-in-time evidence; this module does not infer them or silently substitute a value when absent.

## Boundaries

The initial implementation covers long-only cash-equity target sizing. It does not model derivatives margin, short positions, currency conversion, market impact, stop execution, or broker-specific order validation. ADV is an estimate, not executable liquidity. No policy setting proves a trade is suitable or profitable; historical/synthetic checks demonstrate software behavior only.

## Validation

The F38 workflow runs Ruff, formatting, mypy, F38 behavior tests, the F37/F36/F35/F34/F33 contract regressions, F21/F22 numerical tests, general CI, and Security on Python 3.11 and 3.12.
