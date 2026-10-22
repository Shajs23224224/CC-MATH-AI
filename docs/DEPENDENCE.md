# F23 — Correlation / Dependency

## Purpose

F23 supplies deterministic cross-asset dependence estimators and principal-component factors. Functions operate on aligned arithmetic return series, not price levels, and return immutable typed contracts with canonical asset ordering and auditable sample/window metadata.

## Public API

Import from `quant.dependence`:

- `correlation_matrix(returns, method="pearson")`
- `covariance_matrix(returns)`
- `oas_shrinkage_covariance(returns)`
- `rolling_dependency(returns, window, method="pearson")`
- `pca_factors(returns, n_components=None, standardize=False)`

Correlation methods include Pearson, Spearman (average ranks for ties) and Kendall tau-b. Sample covariance uses the unbiased `n - 1` divisor. OAS shrinkage instead starts with the maximum-likelihood covariance (divisor `n`) and shrinks toward a scaled identity matrix; the estimated intensity is returned in the contract.

## Data and time-window rules

- The input is a mapping from asset symbols to return sequences.
- At least two assets and three observations are required.
- Series must have exactly equal lengths and all values must be finite. Missing values are rejected; there is no implicit row dropping, forward filling or imputation. Callers must align timestamps upstream and should document the timestamp convention.
- Asset symbols are trimmed, uppercased, checked for collisions, then sorted for deterministic matrix axes.
- Rolling results use zero-based, end-exclusive indices: a window with `start_index=10`, `end_index=30` contains observations 10 through 29.
- Correlation matrices have a unit diagonal and bounded coefficients. Pearson/Spearman matrices are checked for positive semidefiniteness; sample/OAS covariance matrices are also checked for positive semidefiniteness. Kendall tau-b is symmetric and bounded but is not rejected solely for a finite-sample non-PSD result.
- Constant series make correlation undefined and cause an explicit error. Covariance permits zero-variance columns.

## PCA / factors

PCA uses centered returns and the sample covariance matrix by default. Set `standardize=True` for PCA of standardized series, equivalent to working from a correlation-scaled input. Components are ordered by descending eigenvalue, and each component sign is fixed by making its largest absolute loading non-negative. The contract records eigenvalues, explained-variance ratios, loadings and historical factor scores. Sign fixing provides repeatability except within degenerate eigenspaces where the eigenspace itself may admit multiple equivalent bases.

## Scope and limitations

- Correlation is not causation, and estimated dependence can change across regimes.
- Pairwise Kendall tau-b is included as a rank-dependence measure, but dependence estimates from finite samples carry uncertainty.
- OAS shrinkage is a regularization estimate, not proof that the target covariance is the true covariance.
- PCA components maximize explained variance; they do not automatically correspond to economically meaningful factors.
- Copulas are left for a later extension because their calibration, tail-dependence choice and validation deserve a separate contract and sample-sufficiency review.

## Validation

The dedicated F23 workflow runs Ruff lint/format, mypy, numerical/contract tests, and regressions for F21 and F22. General CI on the pull request validates Python 3.11 and 3.12, while Security runs dependency audit and static scanning.
