# C-MATH-AI — Software Architecture

**Phase:** F02 — Software Architecture  
**Status:** BASELINE  
**Version:** 0.1.0  
**Depends on:** F01 — `SYSTEM_SPEC.md`  
**Repository:** `Shajs23224224/CC-MATH-AI`  
**Default branch:** `main`

---

## 1. Architectural Objective

F02 converts the F01 system specification into concrete software boundaries that can be implemented, tested and evolved independently.

The architecture must provide:

- strong module boundaries;
- explicit dependency direction;
- stable contracts between modules;
- deterministic quantitative computation;
- replaceable data providers;
- replaceable ML models;
- isolated risk controls;
- reproducible decisions;
- testability without external services;
- paper-trading support before live execution;
- an auditable path from raw data to final eligibility.

The architecture is intentionally modular so that additional algorithms, data providers, models and asset classes can be introduced without rewriting unrelated components.

---

## 2. Architectural Style

C-MATH-AI uses a **layered modular architecture with ports-and-adapters boundaries**.

Conceptually:

```
                ┌─────────────────────────────┐
                │        APPLICATIONS         │
                │ CLI / API / Dashboard / Jobs│
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │       DECISION / ORCHESTR.  │
                │ Signals / Portfolio / Risk │
                └──────────────┬──────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │                DOMAIN / QUANT               │
        │ Contracts / Algorithms / Statistics / ML    │
        └──────────────────────┬──────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │        INFRASTRUCTURE       │
                │ Data / Storage / Providers  │
                │ Execution / External APIs   │
                └─────────────────────────────┘
```

The most important architectural rule is:

> **Higher-level orchestration depends on stable domain contracts, not on specific infrastructure implementations.**

---

## 3. Repository Architecture

The target repository layout is:

```text
CC-MATH-AI/
├── apps/
│   ├── api/
│   ├── cli/
│   └── dashboard/
│
├── core/
│   ├── contracts/
│   ├── config/
│   ├── errors/
│   ├── logging/
│   └── utils/
│
├── data/
│   ├── providers/
│   ├── ingestion/
│   ├── normalization/
│   ├── quality/
│   ├── corporate_actions/
│   └── storage/
│
├── quant/
│   ├── returns/
│   ├── statistics/
│   ├── technical/
│   ├── fundamentals/
│   ├── valuation/
│   ├── timeseries/
│   ├── volatility/
│   ├── stochastic/
│   ├── derivatives/
│   └── registry/
│
├── ml/
│   ├── features/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── calibration/
│   └── evaluation/
│
├── signals/
│   ├── generators/
│   ├── ensemble/
│   ├── meta/
│   └── explainability/
│
├── portfolio/
│   ├── optimization/
│   ├── allocation/
│   └── sizing/
│
├── risk/
│   ├── metrics/
│   ├── constraints/
│   ├── stress/
│   └── governor/
│
├── backtest/
│   ├── engine/
│   ├── execution/
│   ├── costs/
│   ├── metrics/
│   └── validation/
│
├── execution/
│   ├── paper/
│   ├── brokers/
│   ├── exchanges/
│   ├── orders/
│   └── policies/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── numerical/
│   ├── regression/
│   └── property/
│
├── docs/
│   ├── SYSTEM_SPEC.md
│   ├── ARCHITECTURE.md
│   └── decisions/
│
├── scripts/
└── README.md
```

F02 defines the architecture. F03 will initialize the practical repository structure and project tooling.

---

## 4. Module Responsibilities

### 4.1 `core`

The shared kernel of stable contracts.

Responsibilities:

- domain entities;
- value objects;
- enums;
- configuration contracts;
- error taxonomy;
- shared logging abstractions;
- deterministic utility functions.

Examples of domain contracts:

- `Asset`;
- `PriceSeries`;
- `ReturnSeries`;
- `FundamentalSnapshot`;
- `MacroSnapshot`;
- `FeatureVector`;
- `Signal`;
- `Forecast`;
- `RiskMetric`;
- `Portfolio`;
- `Position`;
- `Order`;
- `DecisionRecord`;
- `BacktestResult`.

`core` must remain lightweight and must not import provider-specific, broker-specific or UI code.

---

### 4.2 `data`

The data boundary.

Responsibilities:

- provider adapters;
- ingestion;
- schema validation;
- normalization;
- corporate-action handling;
- quality checks;
- persistence/retrieval interfaces.

Data providers must be hidden behind interfaces.

Example:

```text
MarketDataProvider
    ├── ProviderAAdapter
    ├── ProviderBAdapter
    └── LocalFileAdapter
```

The quant engine must not know which external provider supplied the data.

---

### 4.3 `quant`

The deterministic quantitative engine.

Responsibilities:

- mathematical functions;
- financial formulas;
- technical indicators;
- statistics;
- valuation;
- time-series methods;
- volatility models;
- stochastic models;
- derivative pricing;
- algorithm metadata.

Every algorithm should implement a consistent callable contract.

Conceptual interface:

```text
Algorithm<I, O>
    id
    version
    validate_input(I)
    compute(I, parameters) -> O
    metadata()
```

The implementation language may vary, but the contract must remain explicit.

---

### 4.4 `quant/registry`

The algorithm registry is the index of available quantitative models.

Each registered algorithm must expose:

```text
algorithm_id
name
category
version
input_schema
output_schema
parameters
formula_or_method
data_requirements
frequency_requirements
units
limitations
test_status
```

The registry allows the orchestration layer to discover algorithms without importing each implementation manually.

---

### 4.5 `ml`

The machine-learning boundary.

Responsibilities:

- feature generation;
- dataset construction;
- training pipelines;
- model artifacts;
- inference;
- calibration;
- evaluation;
- model metadata.

ML models consume validated features and emit typed predictions.

Conceptual interface:

```text
Model<I, O>
    model_id
    version
    fit(dataset)
    predict(features) -> Prediction
    metadata()
```

ML code must not bypass data-quality checks.

---

### 4.6 `signals`

The signal intelligence layer.

Responsibilities:

- turning algorithm/model outputs into normalized evidence;
- individual signal generation;
- ensemble aggregation;
- meta-model decisioning;
- explainability.

The signal layer does not place orders.

Conceptual flow:

```text
Algorithm Outputs
      ↓
Normalized Evidence
      ↓
Signal Generators
      ↓
Ensemble
      ↓
Meta Decision
      ↓
DecisionRecord
```

---

### 4.7 `portfolio`

The portfolio construction layer.

Responsibilities:

- asset eligibility;
- allocation;
- optimization;
- position sizing;
- portfolio-level transformations.

The portfolio layer may request a target position, but it cannot bypass the Risk Governor.

---

### 4.8 `risk`

The mandatory risk-control boundary.

Responsibilities:

- risk metrics;
- limits;
- concentration;
- exposure;
- drawdown;
- liquidity;
- stress testing;
- trade eligibility.

The central component is:

```text
RiskGovernor
    evaluate(decision, portfolio, market_state)
        -> RiskDecision
```

Possible output:

```text
ALLOW
REDUCE
BLOCK
```

The Risk Governor is the final internal gate before execution eligibility.

---

### 4.9 `backtest`

The historical simulation and validation engine.

Responsibilities:

- event-driven simulation;
- order lifecycle;
- transaction costs;
- slippage;
- portfolio accounting;
- PnL;
- performance metrics;
- walk-forward validation;
- robustness testing.

Backtest code must use the same decision contracts used by the research pipeline wherever practical.

---

### 4.10 `execution`

The execution boundary.

Execution is split into:

```text
PaperExecution
BrokerAdapter
ExchangeAdapter
```

A common order contract is required so the strategy does not depend on a specific broker.

Execution must accept only decisions that passed the applicable risk policy.

---

### 4.11 `apps`

Application entry points.

Examples:

- REST API;
- CLI;
- dashboard;
- scheduled research jobs;
- batch data jobs.

Applications orchestrate domain services; they should not contain financial formulas.

---

## 5. Dependency Rules

The dependency graph is intentionally one-directional.

### Allowed high-level dependencies

```text
apps
 ↓
signals / portfolio / risk / backtest / execution
 ↓
ml / quant
 ↓
core

data
 ↓
core

execution
 ↓
core + risk contracts

tests
 ↓
any module under test
```

### Mandatory rules

**R01 — `core` has no application/infrastructure dependency.**

```text
core → NOTHING ABOVE CORE
```

**R02 — `quant` may depend on `core`, but not on brokers, UI or concrete data providers.**

**R03 — `ml` may depend on `core`, `quant` outputs and validated feature contracts, but not on broker adapters.**

**R04 — `signals` may depend on quantitative/ML outputs and risk contracts, but never on UI code.**

**R05 — `portfolio` may depend on signal contracts and quantitative/risk metrics, but must not directly call broker APIs.**

**R06 — `risk` must be independently testable and must not depend on dashboard/API implementations.**

**R07 — `execution` must not be called directly by an AI/LLM component.**

**R08 — `execution` must receive risk-approved decisions.**

**R09 — external providers are adapters, never domain contracts.**

**R10 — modules communicate through explicit contracts rather than hidden global state.**

---

## 6. Direction-of-Authority Rule

C-MATH-AI follows this authority chain:

```text
DATA QUALITY
     ↓
QUANT / ML EVIDENCE
     ↓
SIGNAL
     ↓
META DECISION
     ↓
PORTFOLIO
     ↓
RISK GOVERNOR
     ↓
EXECUTION POLICY
     ↓
PAPER / BROKER / EXCHANGE
```

No downstream component may silently override an upstream safety constraint.

In particular:

```text
LLM / AI
   X
   └──> direct broker order
```

is prohibited.

The permitted relationship is:

```text
LLM / AI
   ↓
interpretation / orchestration
   ↓
typed decision request
   ↓
risk validation
   ↓
execution policy
```

---

## 7. Primary Data Flow

The canonical research flow is:

```text
Provider
  ↓
Ingestion
  ↓
Normalization
  ↓
Quality Validation
  ↓
Canonical Data
  ↓
Feature Generation
  ↓
Quant Algorithms
  ↓
Regime Detection
  ↓
ML Inference
  ↓
Signal Ensemble
  ↓
Meta Decision
  ↓
Portfolio Construction
  ↓
Risk Governor
  ↓
Decision Record
```

Paper trading adds:

```text
Decision Record
  ↓
Execution Policy
  ↓
Paper Order
  ↓
Simulated Fill
  ↓
Portfolio Update
  ↓
Audit Log
```

---

## 8. Research Request Lifecycle

A canonical analysis request should follow this sequence:

### Step 1 — Input validation

Validate:

- asset identifier;
- analysis timestamp;
- requested horizon;
- requested frequency;
- required data availability.

### Step 2 — Data retrieval

Retrieve the minimum data required for the requested algorithms.

### Step 3 — Data quality

Reject, warn or downgrade invalid observations.

### Step 4 — Feature calculation

Create deterministic features from the available information timestamp.

### Step 5 — Algorithm execution

Run the requested/eligible algorithms.

### Step 6 — Regime evaluation

Estimate the relevant market/asset regime.

### Step 7 — ML inference

Run only models compatible with the feature/model versions.

### Step 8 — Ensemble

Combine evidence while retaining individual contributions.

### Step 9 — Meta decision

Generate a structured candidate decision.

### Step 10 — Portfolio transformation

Convert an eligible signal into a portfolio-level target.

### Step 11 — Risk evaluation

The Risk Governor returns:

```text
ALLOW / REDUCE / BLOCK
```

### Step 12 — Audit

Persist a complete decision record.

---

## 9. Contract Boundaries

The following contracts are architectural boundaries.

### 9.1 Data contracts

```text
Asset
MarketBar
PriceSeries
FundamentalSnapshot
MacroSnapshot
CorporateAction
DataQualityReport
DataSnapshot
```

### 9.2 Quant contracts

```text
AlgorithmInput
AlgorithmOutput
AlgorithmMetadata
AlgorithmParameters
```

### 9.3 ML contracts

```text
FeatureVector
Dataset
Prediction
ModelMetadata
CalibrationResult
EvaluationResult
```

### 9.4 Decision contracts

```text
Evidence
Signal
EnsembleResult
DecisionRecord
```

### 9.5 Portfolio contracts

```text
TargetAllocation
Position
Portfolio
SizingRequest
SizingResult
```

### 9.6 Risk contracts

```text
RiskMetric
RiskConstraint
RiskEvaluation
RiskDecision
StressScenario
```

### 9.7 Execution contracts

```text
Order
OrderRequest
ExecutionPolicy
Fill
ExecutionReport
```

---

## 10. Immutability and Auditability

Where possible, outputs that affect a decision should be treated as immutable records.

A decision should reference:

```text
data_snapshot_id
feature_set_version
algorithm_versions
model_versions
configuration_version
code_revision
created_at
random_seed (when applicable)
```

The system should prefer append-only decision history over destructive mutation.

---

## 11. State Management

Avoid uncontrolled global mutable state.

State should be explicit:

```text
Request Context
      ↓
Domain Inputs
      ↓
Domain Outputs
      ↓
Persisted Artifacts
```

Caches may be introduced for performance, but cache contents must never become an untracked source of truth.

---

## 12. Synchronous vs Asynchronous Work

### Synchronous operations

Suitable for:

- single-asset calculation;
- individual algorithm execution;
- small decision requests;
- API queries;
- deterministic mathematical functions.

### Asynchronous operations

Suitable for:

- bulk data ingestion;
- historical feature computation;
- large-scale backtesting;
- model training;
- walk-forward evaluation;
- Monte Carlo studies;
- batch portfolio analysis.

The same domain contracts should be usable in both modes.

---

## 13. Storage Architecture

F02 defines storage as an abstraction, not a fixed vendor.

Potential logical stores:

```text
Market Data Store
Fundamental Data Store
Feature Store
Model Registry
Experiment Store
Decision Store
Execution/Audit Store
```

The first implementation can use a local development database/file store, provided the application accesses it through repository interfaces.

Example:

```text
MarketDataRepository
FeatureRepository
ModelRepository
DecisionRepository
ExecutionRepository
```

---

## 14. Configuration Architecture

Configuration must be externalized.

Logical environments:

```text
development
testing
paper
production
```

Configuration categories:

- data provider settings;
- computation defaults;
- algorithm parameters;
- model references;
- risk limits;
- execution policy;
- logging;
- storage;
- feature flags.

Secrets must not be part of source-controlled configuration.

---

## 15. Error Architecture

Errors must be explicit and categorized.

Suggested categories:

```text
ConfigurationError
DataUnavailableError
DataQualityError
ContractValidationError
AlgorithmInputError
AlgorithmComputationError
ModelError
RiskViolationError
PortfolioError
ExecutionError
StorageError
ExternalProviderError
```

Critical numerical errors must fail loudly rather than returning plausible-looking invalid values.

---

## 16. Observability

Every major pipeline stage should expose structured telemetry.

Minimum fields:

```text
request_id
decision_id
asset
timestamp
module
operation
duration_ms
status
error_code
version
```

Decision-critical computations should be traceable across modules using a shared correlation identifier.

---

## 17. Testing Architecture

Testing is organized by scope.

### Unit tests

Test individual algorithms, validators, transformations and contracts.

### Numerical tests

Compare mathematical outputs against trusted reference values.

### Integration tests

Test boundaries such as:

- provider → normalization;
- normalization → quality;
- quant → signal;
- signal → portfolio;
- portfolio → risk;
- risk → paper execution.

### Regression tests

Prevent previously validated behavior from changing unexpectedly.

### Property-based tests

Useful for invariants such as:

- return composition;
- covariance symmetry;
- portfolio weight constraints;
- non-negative volatility;
- split-adjustment consistency.

---

## 18. Determinism Requirements

Deterministic components must return the same result for the same:

- input data;
- parameters;
- configuration;
- implementation version.

Stochastic components must record:

- random seed;
- model version;
- training dataset version;
- hyperparameters.

---

## 19. Performance Strategy

Performance optimization must follow correctness.

Priority order:

```text
Correctness
  ↓
Reproducibility
  ↓
Observability
  ↓
Profiling
  ↓
Optimization
```

Potential future optimizations:

- vectorized numerical operations;
- batch inference;
- caching;
- parallel backtesting;
- asynchronous workloads;
- incremental feature computation.

No optimization may change numerical semantics without tests.

---

## 20. Security Boundaries

Security-sensitive operations include:

- secrets;
- external API credentials;
- broker credentials;
- order submission;
- model artifact loading;
- external data ingestion.

Broker credentials must be unavailable to components that do not need execution privileges.

Application/API clients should interact with execution through a constrained policy layer.

---

## 21. LLM / AI Boundary

An AI/LLM component may:

- select appropriate analytical workflows;
- summarize model outputs;
- explain evidence;
- generate research questions;
- orchestrate approved tools;
- classify unstructured text when validated;
- assist with diagnostics.

The AI/LLM must not be authoritative for:

- numerical formulas;
- raw portfolio accounting;
- risk-limit overrides;
- secret management;
- direct unrestricted order submission.

Any AI-generated action request must become a typed, auditable request before reaching the deterministic decision/risk path.

---

## 22. Versioning Boundaries

Independent versioning is required for:

- system specification;
- algorithm implementation;
- algorithm parameter schema;
- feature definitions;
- datasets;
- ML models;
- risk policies;
- execution policies;
- API contracts.

A decision must be possible to reconstruct using the recorded versions.

---

## 23. API Boundary

The future API should expose services such as:

```text
GET  /assets/{symbol}
GET  /analysis/{symbol}
POST /analysis
GET  /signals/{symbol}
GET  /decisions/{id}
GET  /portfolio
GET  /risk
POST /backtests
GET  /backtests/{id}
POST /paper/orders
GET  /paper/portfolio
```

Exact routes are implementation details for later phases.

The API must call application/domain services rather than implementing business logic itself.

---

## 24. Architectural Invariants

These properties must remain true as the system evolves:

**A01** — Financial calculations are performed by deterministic computational modules.

**A02** — External data providers are hidden behind adapters.

**A03** — No single algorithm has unrestricted decision authority.

**A04** — Risk controls can block/reduce a portfolio action.

**A05** — Execution cannot bypass risk policy.

**A06** — Every decision can be traced to data/model/software versions.

**A07** — Core domain contracts are independent from infrastructure.

**A08** — No secret is required inside the core domain.

**A09** — Backtests account for realistic execution assumptions.

**A10** — New algorithms can be registered without rewriting the decision engine.

---

## 25. Recommended Implementation Order

After F02, the implementation sequence should be:

1. F03 — repository initialization;
2. F04 — mathematical/domain contracts;
3. F05 — configuration and environments;
4. F06 — tests and CI/CD;
5. F07 — market data engine;
6. F08–F12 — fundamental, macro, corporate-action, normalization and data-quality layers;
7. F13 onward — quantitative engine and higher layers.

The architecture is intentionally established before the large algorithm library so that algorithms do not become tightly coupled utilities.

---

## 26. Definition of F02 Completion

F02 is complete when:

- [x] module boundaries are defined;
- [x] target repository architecture is defined;
- [x] dependency direction is defined;
- [x] domain/infrastructure separation is defined;
- [x] algorithm contract is defined;
- [x] ML contract is defined;
- [x] signal/decision boundaries are defined;
- [x] portfolio/risk/execution boundaries are defined;
- [x] data flow is defined;
- [x] storage abstraction is defined;
- [x] configuration boundaries are defined;
- [x] error architecture is defined;
- [x] observability requirements are defined;
- [x] testing architecture is defined;
- [x] AI/LLM authority boundary is defined;
- [x] architectural invariants are defined.

**F02 status: COMPLETE.**

---

## 27. Next Phase

**F03 — Repository Initialization**

F03 will turn this architecture into the actual project skeleton, tooling, documentation entry points, configuration scaffolding and development conventions.

F03 must implement the boundaries established here rather than introducing a competing architecture.
