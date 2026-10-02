# F19 — Time Series

F19 introduces a deterministic econometric layer for temporal models. The implementation is separated from data ingestion and from later portfolio/risk decisions.

## Scope

Implemented model families:

- AR.
- MA.
- ARMA.
- ARIMA.
- SARIMA.
- ARIMAX.
- Exponential smoothing.
- Holt.
- Holt-Winters.
- VAR.
- VECM.

The estimators are exposed through a common typed facade. Underlying model fitting uses statsmodels; critical application logic remains in this repository and model outputs are validated through C-MATH-AI contracts.

## Temporal discipline

F19 does not accept a generic shuffled train/test split. temporal_train_test_split returns chronological train/test partitions, preserving the information boundary.

Fitted models only see the observations supplied to their fit_* function. Forecasts are generated strictly for future steps. ARIMAX additionally requires explicit future exogenous observations; they cannot be inferred from the training sample.

For future walk-forward and OOS work, fitted parameters must remain frozen throughout each test window. F19 therefore provides the model primitive, not an implicit backtest protocol.

## Model notes

### AR / MA / ARMA / ARIMA

These models are represented through the ARIMA state-space implementation with explicit orders. fit_ar, fit_ma and fit_arma are constrained wrappers, while fit_arima accepts (p, d, q).

### SARIMA

fit_sarima adds explicit seasonal (P, D, Q, s) structure. Seasonal period must be at least two.

### ARIMAX

fit_arimax requires a rectangular exogenous design aligned one row per target observation. Future forecasts require a matching future_exog matrix.

### Exponential smoothing / Holt / Holt-Winters

Simple exponential smoothing, additive Holt and additive Holt-Winters variants are provided as deterministic model primitives. F19 intentionally does not tune smoothing parameters through random search.

### VAR / VECM

Multivariate models require a rectangular matrix. fit_var exposes lagged multivariate dynamics. fit_vecm exposes cointegration-rank constrained error correction. Forecast results preserve the complete multivariate matrix in multivariate_values while values exposes the first series as the canonical scalar view.

## Diagnostics

residual_diagnostics reports:

- residual mean and variance;
- residual lag-1 autocorrelation;
- Ljung-Box Q statistic and p-value.

The diagnostics are descriptive evidence, not a claim that a model is correctly specified.

## Contracts

- TimeSeriesForecast: finite future values with explicit horizon and model family.
- TimeSeriesFitSummary: observation count, parameter count, information criteria and convergence flag.
- ResidualDiagnostics: residual moments and serial-correlation diagnostics.
- TemporalSplit: explicit chronological train/test sizes.

## Validation and limitations

- Inputs must be finite and sufficiently long for the selected model.
- Orders and seasonal periods are validated before estimator construction.
- Exogenous data must be rectangular and point-aligned.
- Multivariate inputs must be rectangular with finite observations.
- No silent imputation is performed.
- No model output is converted into a BUY/HOLD/SELL decision.
- No risk limit or future Risk Governor is bypassed.

F19 is the econometric model layer. Walk-forward/OOS evaluation remains a later phase.
