# F14 — Returns & Performance

## Objective

F14 provides deterministic, auditable calculations for investment returns and basic performance measurement.

The implementation consumes normalized price series and simple-return sequences. It does not perform forecasting, portfolio optimization or machine-learning inference.

## Return calculations

### Simple return

R_t = P_t / P_(t-1) - 1

### Log return

r_t = ln(P_t / P_(t-1))

Simple returns are used for wealth compounding and performance metrics. Log returns are available as a separate analytical representation.

### Cumulative return

R_cum = product(1 + R_t) - 1

The implementation accumulates log-growth internally to reduce numerical error.

## Wealth and drawdown

The wealth index starts at a configurable positive initial value and compounds simple returns.

Drawdown is measured relative to the running wealth maximum:

DD_t = W_t / max(W_u for u <= t) - 1

Maximum drawdown is returned as a positive fraction.

## Annualization

F14 requires an explicit periods_per_year argument. This avoids silently assuming a market calendar or trading convention.

Annualized return uses geometric compounding:

R_ann = (1 + R_cum)^(m/n) - 1

where m is periods per year and n is observations.

Annualized volatility uses sample standard deviation multiplied by sqrt(m).

## Risk-adjusted performance

### Sharpe ratio

F14 uses periodic risk-free return supplied explicitly:

Sharpe = ((mean return - risk-free return) / sample standard deviation) * sqrt(m)

### Sortino ratio

Downside deviation is the root-mean-square of returns below the supplied periodic target.

Sortino = ((mean return - target return) / downside deviation) * sqrt(m)

A zero denominator is treated as undefined and raises ValueError rather than returning an arbitrary infinite value.

## Performance summary

PerformanceMetrics includes:

- observation count;
- periods per year;
- total return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- positive-period ratio.

## Scope boundaries

F14 intentionally does not implement:

- hypothesis tests;
- rolling statistical estimators;
- factor regressions;
- autocorrelation analysis;
- time-series models;
- Value at Risk or Expected Shortfall;
- portfolio optimization;
- transaction-cost modeling.

Those responsibilities belong to later phases.

## F14 completion criteria

- [x] Simple returns
- [x] Log returns
- [x] Return series contract
- [x] Cumulative return
- [x] Wealth index
- [x] Drawdown
- [x] Annualized return
- [x] Annualized volatility
- [x] Sharpe ratio
- [x] Sortino ratio
- [x] Positive-period ratio
- [x] Performance metrics contract
- [x] Numerical/reference tests
- [x] Documentation

**Status: IMPLEMENTED**

Next phase: F15 — Statistical Engine.
