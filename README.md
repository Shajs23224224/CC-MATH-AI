# C-MATH-AI

Quantitative research and investment decision-support platform built around deterministic mathematics, statistical models, machine learning, portfolio construction and explicit risk controls.

## Project status

Current phase: **F18 — Valuation**

Completed:
- F01 — System Specification
- F02 — Software Architecture
- F03 — Repository Initialization
- F04–F15 — Data, normalization, mathematics, returns and statistical engine
- F16 — Technical Analysis
- F17 — Fundamental Analysis
- F18 — Valuation

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
- risk enforcement in `risk/`;
- portfolio construction in `portfolio/`;
- simulation in `backtest/`;
- execution adapters in `execution/`.

## Security

Never commit API keys, broker credentials, access tokens or other secrets.

Use environment variables or a secret manager for credentials.

## Disclaimer

C-MATH-AI is a software research and decision-support project. Historical simulation or model output does not guarantee future investment results.
