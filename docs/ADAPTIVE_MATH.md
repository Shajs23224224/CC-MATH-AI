# F28 extension — Adaptive mathematical intelligence

## Purpose and boundary

The adaptive engine combines methods rather than assuming one algorithm fits every dataset. It currently accepts validated, supervised, real-valued regression datasets from the existing MLDataset contract. Classification remains supported by ClassicalMLModel, but it is not silently mixed into the regression discovery workflow.

This is an engineering foundation, not a general artificial mathematician. It discovers and compares concrete candidate models within finite, declared search spaces. It does not claim autonomous scientific understanding, universal algorithm invention or formal theorem proving.

## Public API

~~~
from core.contracts.adaptive_math import AdaptiveMathConfig
from core.contracts.ml import MLDataset
from ml import AdaptiveMathEngine, predict_adaptive_candidate

x_values = tuple(float(value) for value in range(80))
dataset = MLDataset(
    dataset_id="synthetic-y-2x-plus-3-v1",
    source="synthetic exact relation y = 2*x + 3",
    feature_names=("x",),
    target_name="y",
    features=tuple((x,) for x in x_values),
    target=tuple(2.0 * x + 3.0 for x in x_values),
    task="regression",
)
report = AdaptiveMathEngine(AdaptiveMathConfig(random_seed=28)).analyze(dataset)
print(report.selected_candidate.strategy)
print(report.selected_candidate.representation)
print(report.selected_candidate.validation_rmse, report.selected_candidate.test_rmse)
print(predict_adaptive_candidate(report.selected_candidate, ((80.0,), (81.0,))))
~~~

The dataset in this example is synthetic. Its known generating relation permits a controlled test, not a claim that arbitrary unknown formulae can be recovered from finite observations.

## Strategy selection

Rows must be ordered according to the permitted observation chronology before the engine is called. The engine creates contiguous train, validation and test partitions:

1. Fit the reference linear model on the train prefix.
2. Evaluate polynomial models whose feature basis is inside the configured degree and term budget.
3. Generate a seeded population of expression trees and evolve candidates through selection, mutation and subtree crossover.
4. Score candidates on validation RMSE plus a configurable complexity penalty.
5. Select one candidate based on validation only, refit its coefficients on the train-plus-validation development prefix, and evaluate the test tail once.
6. Independently recompute the test RMSE using scalar math.fsum, check output shape/finiteness and return a structured evidence report.

The current evolutionary representation supports feature terminals, numeric constants, +, -, *, /, sin, cos, tanh and abs. Coefficients outside the expression tree are fitted from training data. A candidate with invalid division, non-finite values, excessive magnitude, depth or node count is rejected rather than rewarded.

The leaderboard retains validation evidence; only the selected final candidate carries the final test metrics. Metrics from the test tail never determine which candidate is selected. Because repeated validation-guided search can still overfit validation, the final result remains EXPERIMENTAL.

## Adaptation and inspectability

Adaptation is performed through transparent strategy comparison, not hidden self-modifying production code. The report stores:
- selected strategy and mathematical representation;
- candidate identifier, coefficients or structured expression tree;
- training, validation and final test metrics;
- train/validation/test boundaries and sample sizes;
- seed, evaluations consumed, rejected candidates and generations completed;
- strategy scores, training-prefix-only association patterns and a reason for selection;
- numerical verification checks, known limitations and explicit formal_proof = false.

Patterns are discovered only from the training prefix in this workflow. Pearson/Spearman associations are exploratory signals; they are neither causal findings nor proof of predictive generalization.

Pass the existing JSONLExperimentStore to analyze(..., store=store) to append a versioned record of the configuration and scalar metrics. Raw datasets, executable model objects and credentials are not persisted by this feature. Failed persistence raises an explicit error.

## Safety and resource budgets

ExpressionNode stores structure, not source code. Evaluation uses a closed operator allow-list; it never invokes eval, exec, dynamic imports or a shell. Unknown operators, out-of-range feature indexes, division near zero, non-finite values and excessive magnitudes fail closed. Candidate prediction rejects stored polynomial expansions exceeding the hard 512-term budget.

AdaptiveMathConfig provides explicit controls for:
- random seed;
- maximum polynomial degree and polynomial term count;
- population, generations and maximum candidate evaluations;
- maximum expression-tree nodes and depth;
- mutation and crossover rates;
- dataset row cap and complexity penalty.

The configuration validates a combined upper bound of 50 million candidate-node-observation work units. This is a conservative accounting estimate, not a wall-clock guarantee. The default is CPU-first and requires no GPU or external service.

## Validation and limitations

Behavioral tests use synthetic affine, quadratic and sinusoidal datasets to check strategy evaluation, deterministic seeds, chronology, safe expression prediction, invalid division, report serialization and experiment persistence. These fixtures validate code behavior; they are not scientific benchmarks or financial evidence.

Not implemented here: general program synthesis, unbounded symbolic mathematics, automated theorem proving, causal discovery, reinforcement-learning control, online self-modification, multi-objective Pareto evolution, deep learning or arbitrary model-artifact serialization. The engine is deliberately regression-focused; use the existing classical classifier adapters for classification tasks. A candidate that predicts well on finite holdout data is not thereby a generally correct algorithm or a proven mathematical law.
