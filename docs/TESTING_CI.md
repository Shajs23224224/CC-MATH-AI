# F06 — Testing, CI/CD and Security

## Objective

F06 establishes automated quality gates so every relevant change is checked before it becomes part of the project baseline.

## Test layers

### Unit
Tests individual contracts, configuration rules and deterministic components.

### Integration
Tests interactions between architectural boundaries.

### Numerical
Checks reference calculations and mathematical invariants with explicit tolerances.

### Regression
Reserved for previously validated behavior that must remain stable.

### Property
Reserved for invariants that benefit from generated inputs.

## CI pipeline

GitHub Actions workflow:

```text
Checkout
  ↓
Python 3.11 / 3.12
  ↓
Install project + dev dependencies
  ↓
Ruff lint
  ↓
Ruff format check
  ↓
Mypy
  ↓
Pytest
  ↓
Coverage report
```

The CI workflow runs on pushes to `main`/ `develop` and pull requests targeting `main`.

Coverage is reported from the beginning, but F06 does not enforce an artificial minimum threshold until the implementation has enough production code to justify one.

## Security pipeline

Security checks run on pushes, pull requests and weekly scheduled scans.

They include:

- `pip-audit` for Python dependency vulnerabilities;
- Bandit for static security analysis.

## Dependency automation

Dependabot monitors:

- Python dependencies;
- GitHub Actions dependencies.

## Quality policy

A failed deterministic test, type check, lint check or security audit is a release-quality issue.

No production execution capability should be added while CI is knowingly failing.

## F06 completion

- [x] Unit test automation
- [x] Integration test automation
- [x] Numerical validation suite
- [x] Ruff lint
- [x] Ruff format validation
- [x] Mypy type checking
- [x] Coverage reporting
- [x] Dependency security audit
- [x] Static security scan
- [x] GitHub Actions CI
- [x] Dependency update automation

**Status: COMPLETE**

Next phase: **F07 — Market Data Engine**.
