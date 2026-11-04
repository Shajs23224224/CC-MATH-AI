# D07.2 — Cost-aware execution decision simulation

## Purpose

D07.2 converts the predictions verified by D07.1 into a deterministic, historical
one-session decision simulation. It compares two decision rules for each registered
classifier:

- argmax: the highest-probability class stored by D06.
- neutral_policy: the D06 probability-threshold policy.

D07.2 does not train a model, retune a threshold, or change a prediction. It consumes
the exact audited_predictions.csv produced by D07.1 and verifies the SHA-256 hashes
in audit_manifest.json before calculating any simulated PnL.

## Signal and execution contract

The D04/D06 target assumes that a signal becomes available after the feature session
closes. D07.2 preserves that time boundary:

| Decision | Simulated action |
|---|---|
| UP | Long at the recorded next-session open; exit at that session's close |
| DOWN | Short at the recorded next-session open; cover at that session's close |
| NEUTRAL | Flat; no position and no transaction cost |

Each position is closed on the same session. Consecutive observations therefore do
not overlap positions. The realized execution return is used only after the decision
has been read from the audited prediction row.

For direction s in {-1, 0, +1}, execution return r, active notional N, and all-in
cost c basis points per side:

    GrossPnL = s * r * N

    Costs = 2 * (c / 10000) * N, when s is nonzero; otherwise 0

    NetPnL = GrossPnL - Costs

The active notional is min(configured_notional, equity_before_trade). The default
configuration starts with 100,000 capital units, uses 10,000 notional units per
session, and charges 5 basis points per side (10 bps round trip). These are
configurable research assumptions, not broker quotes.

## Capital and risk accounting

- Each algorithm/strategy pair gets an independent simulated account.
- Notional is fixed per trade, capped by the account's current equity.
- PnL is additive. Profits are not reinvested and returns are not compounded.
- If equity reaches or falls below zero, later positions for that account are skipped.
- A short can lose more than its notional if the underlying rises sharply. The simulator
  records the resulting negative equity and stops that account; it does not assume
  limited short liability or invent a broker margin liquidation price.
- simple_total_return_on_initial_capital is cumulative net PnL divided by initial
  capital. simple_annualized_return is a linear annualization, not CAGR.
- Sharpe and Sortino are calculated from daily net PnL divided by initial capital.
  They are diagnostics under this fixed-notional model, not portfolio-performance
  guarantees.

## Cost assumptions and exclusions

cost_bps_per_side is an all-in per-leg estimate intended to aggregate commissions,
spread and slippage. D07.2 applies it twice for a completed entry/exit. Set the
parameter using conservative assumptions for the intended instrument and venue.

The current model does not estimate or separately apply market impact, exchange taxes,
borrow fees, short-sale rebates, funding, financing, dividends, corporate actions,
partial fills, rejected orders, latency, broker margin rules or intraday liquidation.
The recorded open and close are theoretical prices, not verified fills.

## Usage

Run after a D07.1 integrity audit has completed successfully:

    python -m scripts.simulate_execution_decisions \
      --audit-dir artifacts/execution_policy_audits/<your_d07_1_audit_directory> \
      --output-root artifacts/execution_simulations \
      --initial-capital 100000 \
      --notional-per-trade 10000 \
      --cost-bps-per-side 5 \
      --trading-days-per-year 252

The values shown are defaults for a reproducible baseline. They should be varied in
a sensitivity analysis before interpreting any result. Do not select the most
profitable cost setting after looking at the OOS results.

## Outputs

Each run creates a new timestamped directory under artifacts/execution_simulations/:

- simulated_decisions.csv: one record for every audited OOS row, for both decision
  strategies, including requested action, executed direction, entry/exit prices,
  gross PnL, transaction costs, net PnL, equity and drawdown.
- simulation_metrics.csv: one summary per algorithm and decision strategy.
- simulation_fold_metrics.csv: PnL, costs, counts and win-rate diagnostics per fold.
- simulation_summary.json: configuration, assumptions, limitations and per-strategy
  results.
- simulation_manifest.json: input provenance and SHA-256 hashes/byte counts for
  every simulation output.

The simulation output directory is never overwritten. Generated experiment data
should remain outside Git.

## Tests and CI

    pytest -q tests/unit/test_simulate_execution_decisions.py
    ruff check scripts/simulate_execution_decisions.py tests/unit/test_simulate_execution_decisions.py
    ruff format --check scripts/simulate_execution_decisions.py tests/unit/test_simulate_execution_decisions.py

The D07 workflow runs this test alongside the D07.1 and D04-D06 regression suite
on Python 3.11 and 3.12.

## Interpretation boundaries

D07.2 is a research simulation, not a live-trading system and not evidence of an
executable edge. Its historical OOS folds were already inspected in D03-D06, so they
are not an untouched prospective test. A positive simulated PnL does not establish
future profitability, fill quality, capacity or suitability for capital deployment.
