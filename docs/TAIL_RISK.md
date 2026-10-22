# F22 — Tail Risk

## Purpose

F22 adds auditable tail-loss estimates and deterministic, versioned stress testing. It is a research and risk-decision support layer; it does not send orders and it must not override an F21 Risk Engine decision, especially BLOCK.

## API

The public functions are exposed from `risk.tail`:

- `historical_var_cvar(returns, confidence_level=0.95, horizon=1)`
- `parametric_var_cvar(returns, confidence_level=0.95, horizon=1)`
- `monte_carlo_var_cvar(returns, confidence_level=0.95, horizon=1, simulations=10_000, random_seed=0)`
- `evt_var_cvar(returns, confidence_level=0.99, threshold_quantile=0.90)`
- `evaluate_stress_scenario(position_weights, scenario)` and `stress_test_portfolio(...)`

All estimators accept a one-dimensional series of arithmetic portfolio returns as fractions (for example, 0.01 means +1%). Results use signed losses as fractions of portfolio notional: positive VaR/CVaR means a loss; a negative value means the modelled tail threshold is a gain. The immutable `TailRiskEstimate` records methodology, confidence, horizon, sample size and, for Monte Carlo, simulation count and seed.

## Methodology and limitations

- **Historical VaR/CVaR:** use the empirical quantile of observed rolling horizon losses. The quantile uses the conservative higher order statistic; CVaR is the mean of observed losses at or beyond VaR.
- **Parametric VaR/CVaR:** fit a normal distribution to arithmetic period returns. Mean scales linearly with horizon and standard deviation scales with the square root of horizon. CVaR uses the analytical normal-tail formula.
- **Monte Carlo VaR/CVaR:** draw independent Gaussian period returns using the sample mean and sample standard deviation, then sum each simulated path across the horizon. The seed and simulation count are explicit in the output.
- **EVT:** fit a Generalized Pareto distribution to losses above a Peaks-Over-Threshold threshold. EVT requires at least 200 observations, 20 exceedances, confidence above the selected threshold quantile and a fitted shape below 1; otherwise it refuses to report a potentially invalid expected shortfall.
- **Stress testing:** apply versioned simple-return shocks to signed exposures relative to equity. Missing a shock for any held position fails closed. Results are deterministic and preserve scenario ID/version.

The parametric and Monte Carlo approaches assume independent, normally distributed arithmetic period returns and use an additive P&L approximation over multi-period horizons. They can underestimate fat tails, jumps, changing volatility, dependence or liquidity effects. EVT is sensitive to threshold and sample selection. Stress results are conditional scenarios, not probabilities or forecasts. No estimate guarantees future outcomes.

## Safety and integration

F22 only emits measurements. It does not place orders, resize positions, or call an execution adapter. Consumers may feed its metrics into risk constraints, but F21's hard BLOCK decision must remain authoritative. Missing required stress shocks, invalid data, too few observations, zero variance for normal models and non-finite calculations produce explicit errors rather than silent fallbacks.

## Validation

The dedicated `.github/workflows/f22-validation.yml` workflow runs Ruff lint/format, mypy, numerical and contract tests, then the F21 risk regression tests.
