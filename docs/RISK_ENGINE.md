# F21 — Risk Engine

## Purpose

F21 adds deterministic, typed risk calculations and a reusable constraint evaluator. It does not place orders and does not replace the Portfolio Risk Governor planned for F37.

## Implemented metrics

- **Beta:** sample covariance of asset and benchmark returns divided by sample benchmark variance.
- **Treynor ratio:** annualized arithmetic excess return divided by estimated beta.
- **Calmar ratio:** geometric annualized return divided by maximum peak-to-trough drawdown.
- **Omega ratio:** sum of gains above a periodic threshold divided by the absolute losses below it.
- **Information ratio:** annualized mean active return divided by sample tracking error.
- **Portfolio exposure:** equity, gross and net exposure, gross leverage, net exposure/equity, cash weight, HHI concentration and largest absolute position weight.
- **Liquidity estimate:** position notional divided by average daily dollar volume; estimated liquidation days assume a configured maximum participation rate.
- **Turnover:** gross traded notional divided by average portfolio value.

Sharpe, Sortino, volatility and drawdown primitives already exist in the F14 deterministic performance layer and remain the source of truth. F21 does not duplicate them.

## Constraints and decision semantics

A RiskConstraint references a canonical RiskMetric name and declares a max or min comparison.

- **ALLOW:** no constraint is breached; reduction factor is 1.0.
- **REDUCE:** at least one soft constraint is breached; the default allocation multiplier is 0.5.
- **BLOCK:** any hard constraint is breached, or a required metric is absent; reduction factor is 0.0.

A missing required metric **fails closed**, even if the related constraint is marked soft. Duplicate metric names and duplicate constraint identifiers are rejected to prevent ambiguous evaluations. Violations and triggered identifiers are retained in RiskEvaluation for auditability.

The evaluate_portfolio_risk function calculates portfolio exposures and accepts caller-supplied metrics such as maximum drawdown, beta, liquidity and turnover. Metric timestamps and methodologies travel with the evaluation. The returned result is advisory domain output only; it is not connected to broker execution.

## Assumptions and limitations

- Return-series metrics require paired, finite simple returns. They reject invalid lengths and undefined denominators instead of silently manufacturing values.
- Annualization is explicit through periods_per_year.
- Portfolio equity is cash plus signed position market values and must be positive before exposure ratios are emitted.
- Portfolio positions are assumed to be valued in the portfolio base currency; multi-currency translation is outside F21 scope.
- The liquidity estimate is not an execution guarantee and does not model market impact, spread, halts or changing volume.
- Turnover is gross traded notional divided by average portfolio value; callers must use a consistent convention across comparisons.
- F21 does not implement VaR/CVaR (F22), covariance/correlation infrastructure (F23) or the full risk governor (F37).

## Validation

The F21 workflow runs Ruff lint/format, mypy, numerical reference tests and unit safety tests. The general CI and Security workflows remain required before integrating the branch.
