# F15 — Statistical Engine

## Objective

F15 extends the F13 mathematical foundation with deterministic statistical analytics used by later quantitative, risk, regime and ML phases.

## Scope

- descriptive statistics: mean, median, variance, standard deviation, quartiles and range;
- robust statistics: median absolute deviation and interquartile range;
- distribution shape: skewness and excess kurtosis;
- rank statistics: average-tie rank data and Spearman correlation;
- dependence: lagged autocorrelation;
- rolling analytics: rolling mean, population standard deviation and endpoint z-score.

All functions validate finite inputs, explicit sample-size requirements and parameter domains. No stochastic estimation, external provider or LLM is involved.

## Definitions

Quantiles use linear interpolation over sorted observations. Skewness and excess kurtosis are standardized central moments using population moment denominators. MAD is the median absolute deviation around the sample median. Rolling standard deviation uses the population denominator because each window is treated as the complete local window. Spearman correlation is Pearson correlation applied to average-tie ranks.

## Design constraints

- deterministic output for identical inputs;
- no look-ahead in trailing rolling windows;
- explicit domain validation;
- reuse F13 correlation primitives instead of duplicating Pearson arithmetic;
- immutable Pydantic contracts for reusable summary outputs.

## Completion criteria

- [x] Descriptive statistics
- [x] Robust statistics
- [x] Distribution-shape statistics
- [x] Rank/dependence statistics
- [x] Rolling statistics
- [x] Typed summary contract
- [x] Reference and domain tests
- [x] Documentation

**Status: IMPLEMENTED — validation pending**
