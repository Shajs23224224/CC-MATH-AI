# F13 — Mathematical Core

## Objective

F13 establishes the deterministic numerical foundation used by later quantitative phases. The implementation is deliberately independent of external market providers, language models and execution systems.

The core rule is:

> Critical mathematical and financial calculations are performed by deterministic code. An AI model may later interpret results or orchestrate algorithms, but it does not replace these computations.

## Scope

F13 includes six reusable domains:

1. Algebra and elementary aggregates
2. Foundational descriptive statistics
3. Probability distributions
4. Time-value-of-money formulas
5. Linear interpolation
6. Bounded scalar optimization

### Algebra

Implemented primitives include arithmetic mean, weighted mean, geometric mean, harmonic mean, dot product and discrete compounding.

Input validation rejects empty sequences, mismatched dimensions and non-finite values.

### Statistics

Implemented primitives include population/sample variance, population/sample standard deviation, sample covariance and Pearson correlation.

These are foundational operations. More specialized statistical tests, robust statistics, rolling statistics and time-series analytics remain in F15 and F19.

### Probability

Implemented primitives include the normal probability density function, normal cumulative distribution function and binomial probability mass function.

The binomial implementation uses logarithmic gamma functions for numerical stability rather than directly constructing large factorials.

### Financial formulas

Implemented time-value-of-money primitives include:

- discount factor;
- present value;
- future value;
- ordinary-annuity present value;
- ordinary-annuity future value;
- ordinary-annuity payment.

The functions use discrete per-period rates and explicitly reject rates at or below -100%.

Returns and performance analytics are intentionally deferred to F14.

### Interpolation

The core provides single-segment linear interpolation/extrapolation and piecewise linear interpolation over strictly increasing coordinates.

No smoothing or model-based interpolation is performed.

### Optimization

F13 provides deterministic bounded scalar minimization with golden-section search.

This is a numerical primitive for later model calibration and parameter selection. It does not perform portfolio optimization, position sizing or asset allocation; those belong to F35-F37.

## Determinism and numerical safety

Functions reject non-finite numeric inputs and invalid domains. Reference tests cover known mathematical values and domain errors.

The optimization primitive returns an immutable OptimizationResult containing:

- method;
- estimated optimum;
- objective value;
- original bounds;
- iteration count;
- convergence flag.

This makes the result auditable without coupling the optimizer to application state.

## Dependency boundary

quant.math depends only on the Python standard library and the stable contract layer where an optimizer result is required.

It does not import:

- market data providers;
- storage;
- ML models;
- signal generators;
- portfolio logic;
- execution APIs.

Later phases should consume F13 primitives instead of re-implementing the same arithmetic.

## F13 completion criteria

- [x] Deterministic algebra primitives
- [x] Foundational descriptive statistics
- [x] Probability primitives
- [x] Financial time-value-of-money formulas
- [x] Linear interpolation
- [x] Bounded scalar optimization
- [x] Typed optimization result contract
- [x] Domain validation
- [x] Numerical/reference tests
- [x] Architecture documentation

**Status: COMPLETE**

Next phase: F14 — Returns & Performance.
