# F24 — Cointegration / Statistical Arbitrage

## Purpose

F24 adds validated unit-root and cointegration tests, pair-regression diagnostics, spread statistics and false-discovery-rate correction. The package returns statistical evidence and measurements; it does not generate orders or imply profitability.

## Public API

Import from `quant.cointegration`:

- `adf_test(values, trend="c", max_lags=None)`
- `kpss_test(values, trend="c")`
- `phillips_perron_test(values, trend="c", max_lags=None)`
- `unit_root_diagnostics(values, trend="c", max_lags=None)`
- `engle_granger_test(dependent, independent, trend="c", max_lags=None)`
- `johansen_test(series, det_order=0, k_ar_diff=1)`
- `fit_cointegrating_pair(dependent, independent, dependent_symbol, independent_symbol, max_lags=None)`
- `spread_zscore(spread, window)`
- `benjamini_hochberg(pvalues, alpha=0.05)`

## Unit-root diagnostics

**ADF (Augmented Dickey-Fuller)** and **Phillips-Perron** test the null that a series contains a unit root. The Phillips-Perron implementation uses the `arch` package and a Newey-West long-run variance correction. **KPSS** uses the opposite null: level- or trend-stationarity. Their p-values must be interpreted according to each test's own null; agreement among tests is evidence, not proof. Deterministic terms and lag selection are explicit and recorded in each result.

The ADF/KPSS/Phillips-Perron helpers require at least 20 finite observations and non-constant input. ADF and Phillips-Perron may take an explicit non-negative lag limit; otherwise ADF uses AIC selection and Phillips-Perron uses the package's default lag rule. The KPSS helper uses its automatic lag-selection procedure.

## Cointegration

- **Engle-Granger:** tests the null of no cointegration for a pair using statsmodels' residual-based test and its cointegration-specific critical values. The regression direction matters for the fitted hedge ratio; swapping dependent and independent variables can change its estimate and finite-sample test result.
- **Johansen:** computes trace and maximum-eigenvalue statistics for a multivariate level panel. The contract records the deterministic-order choice and lag differences. The reported rank is the sequential rank at the 5% critical-value level.
- **Pair fit:** ordinary least squares estimates an intercept and hedge ratio, then forms the residual spread. Half-life is estimated from a first-order spread-dynamics regression and is returned only when it implies a finite positive mean-reversion half-life.
- **Spread z-score:** calculates a trailing-window z-score for each window, including the current observation. Constant windows are rejected.

All input series must be aligned to the same timestamps and contain finite observations. The module does not silently drop or impute missing data. Asset labels are normalized, checked for collisions and sorted for multivariate Johansen inputs.

## Multiple testing

Pair searches can examine hundreds or thousands of candidate pairs; unadjusted p-values then overstate evidence. `benjamini_hochberg` returns adjusted p-values and decisions in original input order, applying Benjamini-Hochberg false-discovery-rate control to the supplied family of hypotheses. The chosen hypothesis family and alpha must be kept fixed and documented for a search.

## Leakage and validation rules

A backtest must estimate transformations, hedge ratios, test-selected pairs and thresholds using **training data only**. Freeze these choices during the subsequent validation/test window; do not refit with future observations and retrospectively use the new parameters for earlier trades. Cointegration can break, residual stationarity can change across regimes, and statistical significance does not account for spread execution costs, borrow costs, market impact, liquidity or multiple model-selection steps not included in the tested hypothesis family.

These outputs are research evidence, not forecasts or profitability guarantees. No broker or order-execution integration is part of F24.

## Validation

The dedicated `.github/workflows/f24-validation.yml` workflow runs Ruff lint/format, mypy, numerical/unit tests and F21/F22/F23 regressions. The pull request also runs the repository-wide Python 3.11/3.12 CI and Security workflows.
