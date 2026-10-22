# F26 — Regime Detection

## Purpose

F26 supplies latent-state and sequential change-point diagnostics for market-state analysis. Outputs are typed, versioned and intended for risk/ensemble research. They are not a standalone buy/sell signal and do not change any F21 risk constraint or execution decision.

## Public API

Import from `quant.regimes`:

- `fit_gaussian_hmm(observations, n_states=2, max_iterations=100, tolerance=1e-4)`
- `fit_markov_switching_gaussian(observations, n_states=2, max_iterations=100, tolerance=1e-4)`
- `filter_regime_observations(observations, fitted_model)`
- `cusum_change_points(observations, reference_mean, reference_std, drift=0.5, threshold=5.0)`
- `bayesian_change_point_probabilities(observations, hazard=0.01, ...)`
- `fit_volatility_regime_thresholds(calibration_returns, window=20, low_quantile=0.33, high_quantile=0.67, version=...)`
- `apply_volatility_regimes(returns, thresholds)`

## Gaussian HMM and Markov switching

Both model helpers fit a univariate Gaussian-emission latent Markov chain using deterministic EM initialization: the emissions have regime-dependent means/variances, and a first-order transition matrix captures persistence. State labels are canonically ordered by ascending fitted mean so indices are stable and interpretable across repeated fits.

Results retain likelihood, convergence flag, iteration count, initial probabilities, transition matrix, emission parameters and per-observation probabilities:

- **In-sample filtered probabilities** condition on observations available through that index, but the parameters were fitted on the entire sample passed to `fit_*`; they are not automatically leakage-free for historical evaluation.
- **Smoothed probabilities** condition on the entire supplied sample, including future observations. They are retrospective only and must not be used as historical live signals in a backtest.
- For strict chronological evaluation, fit the model on a training window and pass held-out observations to `filter_regime_observations`. This freezes the fitted parameters and computes only forward probabilities, without refitting or backward smoothing.
- Filtered and smoothed state indices are retained separately.

The current implementation is univariate; multivariate emissions and state-dependent regression designs require a separate validated contract. Labels are statistical states ordered by fitted mean, not semantic facts such as “risk-on” or “risk-off”. State identity and transition estimates can shift across samples; compare fits only after explicit alignment.

## CUSUM

`cusum_change_points` applies a two-sided cumulative-sum detector using a supplied calibration mean and standard deviation, allowance/drift and alarm threshold. Every observation is processed sequentially. The reference parameters should be fit on a training window and frozen before monitoring the next window. Threshold selection is a false-alarm versus delay trade-off, not a probability guarantee.

## Bayesian online change-point probabilities

`bayesian_change_point_probabilities` uses a constant hazard and a conjugate Normal-Inverse-Gamma predictive model. It returns per-time posterior change probabilities and most-likely run lengths. The run-length posterior is truncated at `max_run_length` to bound compute/memory cost. Results depend on the hazard and prior hyperparameters; the probability is conditional on these assumptions and is not a calibrated financial-crisis probability.

## Volatility regimes

Fit thresholds with `fit_volatility_regime_thresholds` using a **training/calibration series only**, then call `apply_volatility_regimes` on the evaluation or monitoring series. Each output uses trailing sample standard deviation and the frozen low/high quantile thresholds; every window is end-exclusive in the reported index metadata. The implementation deliberately separates threshold fitting and application to prevent future-volatility leakage.

## Validation and safety

- Inputs must be finite, one-dimensional and long enough for the selected method; missing values are rejected, not imputed.
- Probability arrays and transition matrices are validated for normalized rows and correct dimensions.
- Calibrated CUSUM and volatility thresholds must be explicit.
- Model outputs are descriptive/statistical, not causal conclusions or profitability guarantees.
- No broker, order placement or override of F21 ALLOW/REDUCE/BLOCK decisions is added.

The dedicated `.github/workflows/f26-validation.yml` runs Ruff lint/format, mypy, F26 tests (including a no-look-ahead regression for held-out filtering) and F21–F25 regressions. General CI runs on Python 3.11/3.12 and Security performs dependency audit plus static analysis.
