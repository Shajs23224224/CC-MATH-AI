# F04 — Mathematical Contracts / Domain Types

## Objective

Define stable, typed contracts shared by the quantitative engine, ML layer, signals, portfolio, risk, backtest and execution modules.

## Contract groups

### Market and data
- Asset
- MarketBar
- PriceSeries
- FundamentalSnapshot
- DataQualityStatus

### Quantitative
- AlgorithmMetadata
- AlgorithmResult

### Decision
- Evidence
- Forecast
- Signal
- DecisionRecord

### Risk
- RiskMetric
- RiskConstraint
- RiskEvaluation
- RiskAction

### Portfolio
- Position
- Portfolio
- TargetAllocation
- SizingResult

### Execution
- Order
- Fill
- ExecutionReport
- OrderSide
- OrderType

### Backtest
- BacktestResult

## Contract rules

1. Contracts use Pydantic models for runtime validation.
2. Domain records are immutable where practical.
3. Enums use explicit serialized values.
4. Required numeric ranges are enforced at the boundary.
5. Version/provenance fields are retained wherever they affect reproducibility.
6. Business logic belongs outside contract models unless it is an invariant of the model itself.
7. External provider and broker implementations consume these contracts rather than redefining equivalent domain structures.

## Important invariants

- probabilities and confidence are constrained to [0, 1];
- risk quantities cannot be negative;
- prices must be positive;
- time-series timestamps must align with values and be strictly increasing;
- market-bar OHLC relationships are validated;
- portfolio allocation weights are bounded to [0, 1] at the individual allocation contract;
- orders require positive quantities;
- backtest capital and costs cannot be negative;
- backtest periods must have an end date after the start date.

## F04 completion

- [x] Market/data contracts
- [x] Fundamental contract
- [x] Quant algorithm contracts
- [x] Signal/forecast/decision contracts
- [x] Risk contracts
- [x] Portfolio contracts
- [x] Execution contracts
- [x] Backtest result contract
- [x] Public contract exports
- [x] Contract validation tests
- [x] Core invariants enforced at model validation boundaries

**Status: COMPLETE**

## Next phase

F05 — Configuration, Secrets and Environments
