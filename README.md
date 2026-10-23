# C-MATH-AI

Quantitative research and investment decision-support platform built around deterministic mathematics, statistical models, machine learning, portfolio construction and explicit risk controls.

## Project status

Current milestone: **F33 — Algorithm Registry (implementation under review; stacked on F32)**

Completed:
- F01 — System Specification
- F02 — Software Architecture
- F03 — Repository Initialization
- F04–F15 — Data, normalization, mathematics, returns and statistical engine
- F16 — Technical Analysis
- F17 — Fundamental Analysis
- F18 — Valuation
- F19 — Time Series
- F20 — Volatility & Stochastic
- F21 — Risk Engine (merged in PR #6)
- F22 — Tail Risk (merged in PR #7; CI and Security passed)
- F23 — Correlation / Dependency: Pearson/Spearman/Kendall matrices, sample/OAS covariance, rolling correlation and PCA factors; dedicated validation, regression tests, general CI and Security passed.
- F24 — Cointegration / Statistical Arbitrage (merged in PR #9; F24/CI/Security passed).
- F25 — Derivatives: Black–Scholes, binomial/trinomial trees, Monte Carlo option pricing, Greeks, implied volatility and versioned volatility surfaces (merged in PR #10; F25, general CI and Security passed).
- F26 — Regime Detection: Gaussian HMM / Markov-switching, CUSUM, Bayesian online change-point probabilities and calibrated trailing-volatility regimes (merged in PR #11; validation workflows passed).
- F27 — Probabilistic Forecasting Engine: expanding-window forecast evaluation, predictive intervals, bootstrap calibration and causal per-origin regime metadata; merged in PR #12 and hardened in PR #13 with CI/Security passed.
- F28 — Classical ML and bounded mathematical discovery (merged and extended): supervised adapters, chronological out-of-sample evaluation, training-prefix association screening, polynomial symbolic regression, expression-tree evolution, validation-based strategy selection and JSONL experiment records. See [adaptive mathematics architecture](docs/ADAPTIVE_MATH.md).
- F29 — Gradient Boosting (integrated in PR #17, merge commit `96fcdb5`): regression/classification adapters, chronological internal validation for tree-count selection, bounded estimator budget, refit on the training prefix, reserved final test tail and auditable metadata. Synthetic behavioral tests validate implementation boundaries, not investment performance. See [F29 technical design](docs/GRADIENT_BOOSTING.md).
- F30 — Unsupervised ML (integrated in PR #18, merge commit `25de38d`): clustering, dimensionality reduction and anomaly detection with training-prefix-only scaling, stability diagnostics and algorithm-honest holdout handling. See [F30 technical design](docs/UNSUPERVISED_ML.md).
- F31 — Deep Learning (integrated in [PR #19](https://github.com/Shajs23224224/CC-MATH-AI/pull/19), merge commit `e5403c6`): optional PyTorch temporal models with chronological train/validation/test splits, training-prefix preprocessing, validation-only early stopping and bounded resource use. Synthetic tests are engineering checks, not evidence of market alpha. See [F31 technical design](docs/DEEP_LEARNING.md).

In implementation / review:
- F32 — Feature Engineering: versioned feature definitions and deterministic, bounded trailing-window transforms for returns, volatility, volume, external factors, risk/regime variables and cross-sectional interactions. Point-in-time availability and lineage controls are documented in [F32 technical design](docs/FEATURE_ENGINEERING.md).
- F33 — Algorithm Registry (under review): typed versioned metadata, duplicate-safe registration, deterministic discovery, lifecycle/test evidence and input/output compatibility checks. The initial manifest is curated; the target of 300+ verified catalog entries is not yet complete. See [F33 technical design](docs/ALGORITHM_REGISTRY.md).

Next planned:
- F34 — Quant Ensemble, after F33 and its dependency chain pass acceptance criteria.

## Architecture

```text
Data
  ↓
Normalization / Quality
  ↓
Quantitative Algorithms
  ↓
Features / Regimes
  ↓
ML + Ensemble
  ↓
Decision
  ↓
Portfolio
  ↓
Risk Governor
  ↓
Backtest / Validation
  ↓
Paper Trading
  ↓
Future Execution Adapters
```

See:
- [Full roadmap F01–F43](docs/ROADMAP_F01_F15.md) — phases F01–F15 and link to detailed phases F16–F43
- [Detailed roadmap F16–F43](docs/ROADMAP_F16_F42.md)
- [F21 Risk Engine](docs/RISK_ENGINE.md)
- [F22 Tail Risk](docs/TAIL_RISK.md)
- [F23 Correlation / Dependency](docs/DEPENDENCE.md)
- [F24 Cointegration / Statistical Arbitrage](docs/COINTEGRATION.md)
- [F25 Derivatives](docs/DERIVATIVES.md)
- [F26 Regime Detection](docs/REGIME_DETECTION.md)
- [F27 Probabilistic Forecasting](docs/PROBABILISTIC_FORECASTING.md)
- [F28 Classical ML and symbolic discovery](docs/CLASSICAL_ML.md)
- [F29 Gradient Boosting](docs/GRADIENT_BOOSTING.md)
- [F30 Unsupervised ML](docs/UNSUPERVISED_ML.md)
- [F31 Deep Learning](docs/DEEP_LEARNING.md)
- [F32 Feature Engineering](docs/FEATURE_ENGINEERING.md)
- [F33 Algorithm Registry](docs/ALGORITHM_REGISTRY.md)
- [SYSTEM_SPEC.md](SYSTEM_SPEC.md)
- [Architecture](docs/ARCHITECTURE.md)

## Development

Recommended Python version: **3.11+**

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run checks:

```bash
make test
make lint
make format-check
```

Run the smoke test directly:

```bash
python -m pytest
```

## Repository layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete design.

The repository intentionally keeps:
- financial calculations in `quant/`;
- domain contracts in `core/`;
- external providers behind `data/` adapters;
- model training/inference in `ml/`;
- decision logic in `signals/`;
- risk metrics and constraint evaluation in `risk/`;
- portfolio construction in `portfolio/`;
- simulation in `backtest/`;
- execution adapters in `execution/`.

## Security

Never commit API keys, broker credentials, access tokens or other secrets.

Use environment variables or a secret manager for credentials.

## Disclaimer

C-MATH-AI is a software research and decision-support project. Historical simulation or model output does not guarantee future investment results.
