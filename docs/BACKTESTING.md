# F39 — Deterministic Backtesting Engine

## Status and scope

F39 introduces a deterministic, event-driven baseline simulator for signed integer positions. It is a research tool only: it does not connect to a broker or send orders. The first version uses bar data, market-style fills at the next bar open, commission basis points and adverse slippage basis points.

## Temporal contract

A target position keyed by timestamp is considered known only after that bar has closed. It is queued and, if it differs from the current position, executed at the next bar's open. A signal on the final bar is not executed because there is no later open. Bars must have strictly increasing timestamps and valid positive OHLC prices. Signal timestamps must refer to bars in the input series.

## Accounting and metrics

The simulator tracks cash, signed units, mark-to-close equity, fills, commissions, slippage cost, turnover, total return, maximum drawdown, Sharpe and Sortino ratios. Commission is charged on fill notional. Slippage worsens the fill price in the direction of the trade. Ratios use periodic equity returns and the configured periods-per-year; they are descriptive estimates, not forecasts. Zero volatility or zero downside deviation returns 0 for the respective ratio rather than infinity.

## Known limitations

- This baseline does not yet model partial fills, queue position, limit/stop orders, market impact, financing, borrow fees, dividends, splits, taxes or currency conversion.
- Signed units allow short exposure; margin, borrow availability and short-sale constraints are not enforced by this first engine. Do not use it as a production risk-control substitute.
- Signals are supplied externally; this release does not yet wire F35/F36/F37/F38 automatically into the event loop. Integrations must fail closed on blocked or missing risk evidence.
- Data quality checks cannot prove absence of survivorship bias or corporate-action errors. Use point-in-time data and a separate out-of-sample validation process.
- Synthetic tests verify accounting and temporal invariants, not strategy profitability or future performance.

## Minimal example

```python
from datetime import datetime, timezone
from backtest.engine import BacktestConfig, Bar, run_backtest

bars = [
    Bar(datetime(2025, 1, 1, tzinfo=timezone.utc), 100, 100, 100, 100),
    Bar(datetime(2025, 1, 2, tzinfo=timezone.utc), 101, 101, 101, 101),
]
result = run_backtest(
    bars,
    {bars[0].timestamp: 1},  # target one unit after the first close
    BacktestConfig(initial_capital=10_000, commission_bps=1, slippage_bps=2),
)
print(result.total_return, result.max_drawdown, result.fills)
```
