# F41 — Monte Carlo and Adversarial Robustness

## Objective and boundaries

F41 quantifies sensitivity of a deterministic F39 backtest to return-sequence uncertainty and declared adverse assumptions. It is a research diagnostic, not proof of alpha, an estimator of the probability of future market outcomes, or permission to execute trades. Run it after F40 walk-forward/OOS validation and keep strategy/data versions fixed.

## Public API

- **run_robustness(bars, targets_by_timestamp, ...)** executes one baseline, configured bootstrap simulations, and each explicitly named stress scenario.
- **run_monte_carlo(backtest_result, config)** builds a terminal-return and max-drawdown distribution from observed positive-equity mark-to-market returns.
- **stress_correlation_matrix(returns_by_asset, shock)** produces baseline and stressed Pearson correlation matrices from synchronized, finite, non-constant return series.

## Monte Carlo assumptions

MonteCarloConfig versions the simulation count, seed, confidence level, sampling method and block size. The iid method samples period returns independently with replacement. The moving_block method samples contiguous blocks from the observed sequence, up to the configured block size, until the original length is reached. A local random.Random instance avoids modifying global RNG state.

The report provides the mean and median simulated return, lower/upper quantiles, loss frequency, lower-tail expected shortfall, and the 95th percentile of maximum drawdown. A simulated equity path is floored at zero if a period return implies total loss or worse; it cannot recover after bankruptcy. The baseline must have a strictly positive equity curve. These simulations resample the existing strategy return stream: they do not retrain the model, rerun the strategy on generated bars, model order-book liquidity, or correct selection bias.

## Adversarial scenarios

A ScenarioSpec has a unique, non-empty name and explicit parameters:

- volatility_scale: scales bar-to-bar OHLC deviations from the preceding transformed close (1.0 preserves the path).
- price_shock_bps and shock_start_index: apply one persistent price-level shock from the chosen index, creating a gap.
- extra_commission_bps and extra_slippage_bps: increase F39 transaction costs additively.
- execution_delay_bars: shifts signal timestamps forward; signals shifted beyond available data are dropped rather than filled at an invented price.

Every scenario re-runs F39 and reports return/drawdown deltas, Sharpe, trade count and modeled costs against the same baseline. Invalid transformed prices and invalid stressed slippage fail explicitly. The aggregate stress_degradation_fraction counts scenarios with a lower return than baseline. fragility_score is an explicitly heuristic diagnostic: 50% Monte Carlo loss frequency plus 50% of the declared scenarios that underperform baseline. It is not a calibrated probability and should never be interpreted as a risk limit.

## Correlation stress

Returns for at least two assets must be aligned in length, finite and non-constant. shock lies in [-1, 1]. Positive shocks blend the observed matrix toward +1 off-diagonal correlation. Negative shocks blend toward the minimum feasible equicorrelation (-1/(n-1)); zero leaves the matrix unchanged. These targets are positive semidefinite and convex blending preserves that property up to numerical rounding. Results include input dimensions, both matrices and a SHA-256 fingerprint.

**Important boundary:** F39 accepts a single OHLC stream and signed units; F41's correlation stress is consequently a matrix diagnostic, not a multivariate portfolio PnL simulation. Applying portfolio weights and asset-level stressed paths requires a future multi-asset accounting contract.

## Reproducibility and limitations

- Seeds, simulation counts, bootstrap assumptions, scenario parameters, synchronized correlation data, baseline metrics and scenario outputs contribute to a canonical SHA-256 fingerprint.
- Use stable data snapshots and a strategy version in upstream workflows; a fingerprint makes this experiment record traceable, but does not prove inputs are economically correct or the strategy is free from look-ahead.
- Stress scenarios are specified by the caller. Results can be misleading when shocks, costs, block sizes or the number of simulations are chosen after looking at favorable outcomes. Record rejected and exploratory scenarios too.
- Synthetic tests establish calculation/contract behavior; they do not demonstrate investment performance or production execution quality.

## Example

    from backtest.robustness import MonteCarloConfig, ScenarioSpec, run_robustness

    report = run_robustness(
        bars,
        target_units_by_timestamp,
        monte_carlo_config=MonteCarloConfig(
            simulations=2000, seed=41, method="moving_block", block_size=10
        ),
        scenarios=(
            ScenarioSpec(
                name="higher-friction",
                extra_commission_bps=5.0,
                extra_slippage_bps=10.0,
                execution_delay_bars=1,
            ),
            ScenarioSpec(
                name="volatility-doubles-and-gap-down",
                volatility_scale=2.0,
                price_shock_bps=-1500.0,
                shock_start_index=500,
            ),
        ),
    )
    print(report.monte_carlo.probability_of_loss)
    print(report.monte_carlo.expected_shortfall)
    print(report.fragility_score, report.fingerprint)


## Integration and release gate

F41 is integrated only after the PR targets `main` and the dedicated F41 workflow, general CI, Security and relevant upstream regression workflows all pass on the final pull-request head. The integration sequence is F39, then F40, then F41; each dependent PR must be validated after retargeting. Monte Carlo and stress outputs are research diagnostics only and do not authorize execution or replace F38 risk approval.
