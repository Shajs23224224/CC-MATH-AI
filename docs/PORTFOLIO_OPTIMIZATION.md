# F36 — Portfolio Optimization

## Purpose and place in the system

F36 converts the typed, timestamped F35 decision outputs into a long-only target portfolio. It consumes a matching F23 covariance matrix, a common forecast horizon, explicit asset/sector constraints and a versioned optimizer policy, then emits a validated `PortfolioOptimizationResult` containing the target `TargetAllocation` rows, solver status, constraint diagnostics and a replay fingerprint.

F36 creates target weights only. It does not translate them into share quantities, submit orders, or override F21 Risk Engine / Risk Governor policy. F37 owns position sizing; risk approval and execution eligibility remain downstream.

## Public API

Import `PortfolioOptimizationConfig`, `PortfolioOptimizationRequest`, `PortfolioAssetInput` and `optimize_portfolio` from `portfolio` (or `portfolio.optimization`).

A request provides:

- One F35 `DecisionOutput` for each asset, current portfolio weight and an optional eligibility / sector / min-max weight constraint.
- An F23 `DependenceMatrix` of kind covariance, sorted in exactly the same symbol order as the decisions.
- Portfolio `as_of`, covariance `as_of`, covariance snapshot ID, and an explicit common horizon.
- Optional Black-Litterman view matrix, view returns, view confidences and equilibrium weights.
- A versioned config naming the algorithm and its parameter values.

The request rejects duplicate or misordered assets, correlation matrices supplied in place of covariance, future decision/covariance timestamps, non-finite expected return/risk, mismatched horizons, invalid view dimensions and current long-only weights greater than the portfolio budget. Covariance is regularized only when required, and the ridge amount is reported in the result.

## Implemented methods

| Method | Objective / procedure |
|---|---|
| `equal_weight` | Minimize squared distance from equal weights under all hard constraints. |
| `mean_variance` | Maximize forecast return net of estimated transaction cost minus a risk-aversion-scaled variance penalty. |
| `minimum_variance` | Minimize portfolio variance, with an optional turnover regularizer. |
| `maximum_sharpe` | Maximize net excess return over portfolio volatility; risk-free rate uses the request horizon's units. |
| `risk_parity` | Minimize squared distance between each asset's normalized variance contribution and an equal-risk target. |
| `inverse_volatility` | Constrained projection toward weights proportional to inverse volatility. |
| `hrp` | Hierarchical Risk Parity with correlation distance, deterministic single-linkage ordering, recursive bisection and inverse-variance cluster risk. |
| `black_litterman` | Compute equilibrium prior returns from risk aversion, covariance and market weights; update them with confidence-weighted views, then run mean-variance optimization. |
| `maximum_diversification` | Maximize weighted asset volatility divided by portfolio volatility. |

The nine methods use SciPy's deterministic SLSQP routine where iterative optimization is needed. The result differentiates `OPTIMAL` (solver convergence), `FEASIBLE` (valid constraints without a solver convergence claim), `INFEASIBLE` (known contradictory bounds / eligibility), and `FAILED` (no accepted feasible candidate found). An attempted solver failure is not represented as an optimal allocation.

## Eligibility policy

- Caller-ineligible assets, F35 abstentions and F35 `SELL` signals receive zero target weight.
- F35 `HOLD` may retain an existing position, but its target is capped at the current weight; HOLD with no existing position is excluded.
- If an excluded asset also has a positive minimum weight, the request is reported as infeasible rather than silently ignoring the user's constraint.
- The optimizer fully invests the modeled target universe. Cash is treated as an implicit part of the current portfolio for turnover accounting, but no target cash allocation is emitted. If the bounds or active-universe policy cannot allocate 100%, the result must not be consumed as a valid target.

These are explicit long-only policy defaults, not universal investment rules.

## Constraints, trading costs and audit trail

Supported hard controls:

- Per-asset minimum and maximum target weights.
- Sector maximum weights.
- Maximum one-way turnover, with implicit current cash deployment included.
- Maximum Herfindahl concentration index (HHI).
- Fully invested weights summing to one.

Each accepted solution carries per-constraint measured values, limits, pass/fail state and explanations. The estimated transaction-cost fraction applies configured basis points to the absolute changes in risky-asset weights; reported one-way turnover is half that amount plus the implied cash deployment change. Cash deployment contributes to turnover but is not charged a commission. Return-based objectives subtract estimated trading costs; all methods report costs and can use the explicit turnover penalty and turnover limit. A cost estimate is only as good as its supplied basis-point assumption.

Every result records the chosen method/config version, forecast and covariance snapshot IDs, horizon, F35 decision fingerprints, solver message/status/iteration count, covariance ridge, expected portfolio return, volatility, net Sharpe diagnostic, turnover, cost estimate, concentration and a canonical SHA-256 request/config fingerprint.

## Units and temporal safety

F36 does not annualize returns or covariance implicitly. F35 expected returns, F23 covariance entries and configured risk-free rate must use compatible units and the exact same horizon. The covariance timestamp must be no later than portfolio `as_of`, and every decision timestamp must also be no later than `as_of`. The contract records timestamps/snapshots; upstream data systems remain responsible for proving the full sample contents and corporate-action handling.

Black-Litterman view returns use those same horizon units. View confidence must be in `(0, 1]`; equilibrium weights must be nonnegative and sum to one. Redundant or ill-conditioned views are handled through a pseudoinverse, and posterior values must remain finite.

## Failure behavior and boundaries

Known simple bound contradictions return `INFEASIBLE` with all target weights set to zero and no expected-return claim. A solver result with breached constraints or non-finite weights is `FAILED`; consumers must only apply allocations when status is `OPTIMAL` or `FEASIBLE` and all constraint diagnostics pass.

F36 cannot guarantee the input expected returns, covariance or Black-Litterman views are statistically accurate; cannot guarantee future diversification or performance; and does not include short selling, leverage, factor constraints, lot-size rounding, tax/market-impact models, or cash targets in this phase. Costs are estimates, not execution quotes. F21/F22 risk assessment, F37 position sizing, backtesting/walk-forward validation and paper-trading checks remain separate.

## Validation plan

The F36 workflow runs Ruff, Ruff format, mypy, behavioral tests for all nine methods, bound/sector/turnover/concentration controls, exclusions, infeasibility, horizon/leakage checks and fingerprint reproducibility. It also runs F34/F33 and F23/F21 regressions. General CI and Security must pass on the PR head. Synthetic unit tests are software validation, not evidence of market alpha or investment suitability.
