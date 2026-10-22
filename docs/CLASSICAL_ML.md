# F28 — Classical ML, pattern discovery and bounded symbolic search

## Scope and audit decisions

The baseline was the main branch at commit b07b3c421b5a0d12bf2e0e0b2fd14b5b1853ae0d, after the F27 causal walk-forward hardening. The repository already contains deterministic NumPy/SciPy mathematical modules and an F27 expanding-window evaluator. The ml package existed only as placeholder directories, and no F28 dataset/model contracts or scikit-learn dependency were present.

F28 adds a supervised model layer, exploratory association records, a bounded polynomial symbolic-regression search, and an append-only experiment ledger. It reuses Pydantic, NumPy, SciPy and the F27 temporal-discipline principles. It does not duplicate F27 probabilistic forecasting, and it does not include clustering, gradient boosting or deep learning, which remain later phases.

## Classical ML API

Create a validated, ordered dataset and fit a registered algorithm:

~~~python
from core.contracts.ml import MLDataset, MLTrainingConfig
from ml import ClassicalMLModel

dataset = MLDataset(
    dataset_id="toy-linear-v1",
    source="synthetic example",
    feature_names=("x",),
    target_name="y",
    features=tuple((float(x),) for x in range(60)),
    target=tuple(3.0 * x - 2.0 for x in range(60)),
    task="regression",
)
model = ClassicalMLModel(MLTrainingConfig(algorithm="ridge", task="regression", random_seed=28))
evaluation = model.fit(dataset)
print(evaluation.metrics)
print(model.predict(((61.0,), (62.0,))).predictions)
~~~

The model catalogue is explicit and task-scoped:
- Regression: linear regression, Ridge, Lasso, Elastic Net, SVR, KNN regressor, decision tree regressor and random forest regressor.
- Classification: logistic regression, SVC, KNN classifier, decision tree classifier and random forest classifier.

Preprocessing is encapsulated in a scikit-learn pipeline and is fitted only on the training prefix. The holdout is the last test_fraction of rows; row order is preserved. Regression reports MAE, RMSE and R². Classification reports accuracy, macro precision, macro recall and macro F1. Supported classifiers expose class probabilities when their estimator supports predict_proba.

MLModelMetadata stores the estimator name, controlled seed, scikit-learn version, training sample count, feature names, resolved scalar parameters and a SHA-256 fingerprint of the training prefix. It does not serialize an executable model object. Parameter names must exist on the allow-listed estimator; random_state is controlled by random_seed, and parallelism is kept at one worker for deterministic CPU-first behavior.

This is model fitting and holdout evaluation, not automated hyperparameter selection. The test tail is not used for model selection. For temporal or market data, the input row order must already represent the permitted observation order; a split cannot repair misordered or point-in-time-invalid source data.

## Pattern discovery

discover_patterns(dataset, min_abs_correlation=0.7) records strong Pearson associations, Spearman rank correlations and the absolute difference between two chronological half-window Pearson correlations. Each record includes an explicit association-only interpretation, a hypothesis, dataset identity and limitations.

A reported association is not a causal finding, a statistical significance claim or a guarantee of predictive generalization. Thresholding and half-window stability are screening diagnostics; multiple testing and independent validation remain necessary for scientific claims.

## Bounded symbolic regression

discover_polynomial_relation(dataset, max_degree=3, max_terms=128) constructs polynomial feature bases using authorized scikit-learn primitives, fits coefficients from data and compares candidate degrees on a chronological validation segment. The final 20% tail is held out from degree selection and used for final reporting. The grammar and number of terms are bounded before expansion. The result stores the chosen basis terms, coefficients, intercept, complexity, validation RMSE, test RMSE/MAE/R², split boundaries and limitations.

predict_polynomial_relation evaluates the stored representation through polynomial-feature expansion and numeric matrix multiplication. It never evaluates candidate source strings, uses eval/exec, imports generated code, or gives candidate computations filesystem, network or process access.

This is deliberately limited symbolic regression: it searches polynomial bases up to degree four, not arbitrary mathematical expressions. It may identify a compact relation compatible with the observed sample, but it cannot prove that the expression is unique or the true generating law. The returned candidate is marked EXPERIMENTAL even when test error is very small.

## Experiment memory

JSONLExperimentStore appends and recovers validated, versioned ExperimentRecord values in JSON Lines format. Invalid persisted records fail with a line-specific error. The store intentionally does not pickle estimators or reload executable artifacts. Records should contain provenance, configuration, metrics and limitations rather than raw confidential data or credentials.

## Errors and resource limits

Input/schema violations raise MLInputError or Pydantic validation errors; estimator failures raise MLTrainingError; prediction failures raise MLPredictionError; persistence failures raise ExperimentPersistenceError. Unknown algorithms/tasks and unknown estimator parameters are rejected. Polynomial degree is limited to 1–4 and the default candidate-basis budget is 128 terms (hard maximum 512). No arbitrary generated code is executed.

## Tests and limitations

The F28 test suite uses synthetic data with known affine, classification and association behavior. It verifies chronological boundaries, repeatability under a fixed seed, probability normalization, budget rejection, symbolic holdout error, candidate reuse and JSONL round-trip/error handling. These synthetic tests are behavior checks, not investment benchmarks.

Not implemented in F28: hyperparameter search, probability calibration, online learning, causal discovery, deep learning, unsupervised learning, arbitrary program synthesis, multi-objective evolutionary search, formal proof, model-artifact serialization or a graphical interface. Those require separate designs and validation in later phases.
