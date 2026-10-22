# C-MATH-AI

Quantitative research and investment decision-support platform built around deterministic mathematics, statistical models, machine learning, portfolio construction and explicit risk controls.

## Project status

Current phase: **F26 — Regime Detection**

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

Active:
- F26 — Regime Detection: Gaussian HMM / Markov-switching, sequential CUSUM, Bayesian online change-point probabilities and calibrated trailing-volatility regimes; implementation and CI validation in progress.

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
- [Full roadmap F01–F42](docs/ROADMAP_F01_F15.md) — phases F01–F15 and link to detailed phases F16–F42
- [Detailed roadmap F16–F42](docs/ROADMAP_F16_F42.md)
- [F21 Risk Engine](docs/RISK_ENGINE.md)
- [F22 Tail Risk](docs/TAIL_RISK.md)
- [F23 Correlation / Dependency](docs/DEPENDENCE.md)
- [F24 Cointegration / Statistical Arbitrage](docs/COINTEGRATION.md)
- [F25 Derivatives](docs/DERIVATIVES.md)
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
