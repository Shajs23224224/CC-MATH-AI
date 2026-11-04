# D07.2 — Cost-aware execution decision simulation

## Objective

D07.2 simulates the decisions contained in a successfully verified D07.1 audit
package. It compares the two persisted D06 decision rules for every classifier:

- argmax: the maximum-probability class.
- neutral_policy: the D06 threshold rule that can abstain to NEUTRAL.

It does not train models, adjust thresholds or choose a cost scenario based on
performance. The default run evaluates a fixed cost grid of 0, 5, 10 and 20 basis
points per side.

## Signal and fill assumptions

D04 defines the signal as available after the feature session closes. The simulator
enters at the recorded next-session open and exits at the same session's close:

| D06 class | Simulated position |
|---|---|
| UP | Long |
| DOWN | Short |
| NEUTRAL | Flat; no position and no transaction cost |

Positions last one session. Consecutive signals therefore do not overlap in the
supplied daily execution dataset. The recorded prices are theoretical execution
prices, not verified market fills.

For direction s in {-1, 0, +1}, realized gross open-to-close return r, active
notional N, and all-in transaction cost c basis points per side:

    Gross PnL = s * r * N
    Round-trip cost = 2 * (c / 10000) * N, when s != 0
    Net PnL = Gross PnL - Round-trip cost

NEUTRAL has zero PnL and zero costs. Each completed entry/exit pays the configured
cost once on entry and once on exit.

## Capital model

- Every algorithm, decision rule and cost scenario receives an independent account.
- Initial capital defaults to 100,000 units.
- Fixed notional defaults to 10,000 units per trade.
- Active notional is capped at the account's current equity.
- PnL is additive; profits are not reinvested and returns are not compounded.
- If equity reaches or falls below zero, the account stops opening new positions.
- A short can lose more than its notional if the underlying rises sharply. The model
  records negative equity and stops that account; it does not invent a broker margin
  liquidation price.

These are explicit research assumptions, not a recommendation for position size.

## Cost sensitivity

The default per-side grid is:

| Scenario | Cost per side | Round trip |
|---|---:|---:|
| Zero-cost reference | 0 bps | 0 bps |
| Low | 5 bps | 10 bps |
| Medium | 10 bps | 20 bps |
| High | 20 bps | 40 bps |

The per-side cost is an all-in estimate intended to aggregate commissions, spread and
slippage. The simulator does not independently estimate market impact, short borrow
fees/availability, funding, financing, dividends, corporate actions, taxes, partial
fills, latency or broker margin/liquidation rules. Use realistic, conservative costs
for the instrument and venue being studied. Do not select the most profitable
scenario after inspecting the OOS results.

## Metrics and outputs

Each timestamped run under artifacts/execution_simulations/ contains:

- trade_ledger.csv: one row per algorithm, policy, cost scenario and OOS session,
  including signal, executed direction, theoretical prices, gross PnL, transaction
  costs, net PnL, equity and drawdown.
- strategy_metrics.csv: trade count, long/short/neutral count, exposure, gross/net PnL,
  total costs, simple return, linear annualization, volatility, Sharpe, Sortino,
  maximum drawdown, win rate, profit factor, turnover and break-even cost.
- fold_metrics.csv: PnL, costs, trades and continuous account equity at each fold
  boundary.
- cost_sensitivity.csv: comparable compact results across scenarios.
- simulation_summary.json: configuration, assumptions, limitations and per-run
  provenance.
- simulation_manifest.json: input identities and output hashes/sizes.

Sharpe and Sortino use daily net PnL divided by initial capital. Annualized return is
linear (not CAGR). The break-even cost is an aggregate, not a forecast of achievable
execution quality.

## Usage

    python -m scripts.simulate_execution_decisions \
      --audit-dir artifacts/execution_policy_audits/<d07_1_audit> \
      --output-root artifacts/execution_simulations \
      --initial-capital 100000 \
      --notional-per-trade 10000 \
      --cost-scenarios-bps-per-side 0,5,10,20 \
      --trading-days-per-year 252

The simulator refuses audit packages with missing/failed checks, mismatched manifests,
inconsistent algorithms/folds, invalid dates, non-finite prices/returns, invalid
probabilities or decisions that cannot be reproduced.

## Interpretation boundary

D07.2 is a historical decision simulation, not live/paper broker execution and not
proof of market alpha. The historic OOS folds were already inspected in D03–D06 and
are not an untouched prospective test. Positive simulated PnL does not establish
future profitability, fill quality, capacity or suitability for capital deployment.

See D07_EXECUTION_DECISION_AUDIT.md for the upstream artifact-verification stage.
