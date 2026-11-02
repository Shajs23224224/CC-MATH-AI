# Training Dashboard — TD02: Instrumentation of training engines

## Status and scope

TD02 adds optional progress callbacks to the existing training APIs. When a caller supplies
a callback, the engines emit typed `TrainingProgress` observations. When no callback is
supplied, the previous call signatures and model behavior remain compatible.

The callback contract belongs to `core/contracts`; it does not import the Dashboard, a web
framework, a browser, PyTorch or scikit-learn. The existing model engines remain responsible
for numerical work. The callback only receives information already calculated by the engine.

TD02 does not add the HTTP backend, SSE transport, graphical interface or central training
command. Those are later phases.

## Public progress contract

`core/contracts/training.py` defines:

- `TrainingProgress`: immutable stage, percentage/step counters, numeric metrics,
  scalar attributes and an optional message.
- `TrainingProgressCallback`: callable accepting one `TrainingProgress` value.

Every engine keeps telemetry optional:

```python
result = model.fit(dataset, progress_callback=callback)
```

The callback is not invoked unless explicitly supplied.

## Connecting a model to the TD01 recorder

```python
from apps.dashboard.event_store import TrainingEventStore
from core.observability.training import TrainingEventRecorder

store = TrainingEventStore("artifacts/training/events.jsonl")
recorder = TrainingEventRecorder(event_sink=store.append)

recorder.create_run(
    name="Ridge baseline",
    algorithm="ridge",
    run_id="ridge-001",
    metadata={"dataset_id": "prices-v1", "seed": 28},
)
recorder.start_run("ridge-001", stage="training")

callback = recorder.progress_callback("ridge-001")
evaluation = model.fit(dataset, progress_callback=callback)

recorder.complete("ridge-001")
```

The run must be started before the callback is used. The caller owns the lifecycle and
should transition the run to FAILED when an exception aborts the training operation.
Persistence failures raised by the recorder's event sink are not silently discarded.

## Instrumented engines

### F28/F29 — Classical ML and Gradient Boosting

`ClassicalMLModel.fit(dataset, progress_callback=None)` reports:

- `model_fit`: estimator fit completed on the chronological training prefix.
- `evaluation`: final holdout metrics after fitting.

For Gradient Boosting, `_select_gradient_boosting_iterations` emits one
`boosting_validation` update per tested estimator count. Regression includes validation
MSE and best validation MSE; classification includes validation log loss and best log loss.
Attributes include best iteration and stale-iteration count. The final selected estimator
count is also reported.

### F28 — Adaptive mathematical discovery

`AdaptiveMathEngine.analyze(dataset, progress_callback=None)` reports:

- chronological data split sizes;
- linear, polynomial and evolutionary candidate evaluation;
- candidate status, strategy, degree/complexity and validation RMSE;
- generation completion and candidate/rejection counts;
- final candidate domain/numerical verification;
- final selected-candidate evaluation.

Candidate-search events expose validation metrics and selection scores only. They do not expose
test metrics. Test metrics are emitted only in the terminal `completed` event, after candidate
selection and final safety verification. The test tail remains excluded from candidate ranking.

### F30 — Unsupervised ML

`UnsupervisedMLModel.fit(features, feature_names, progress_callback=None)` reports:

- chronological split and training/holdout counts;
- training-prefix-only scaling;
- estimator fit;
- cluster stability ARI for each configured resampling repeat;
- operation-specific diagnostics (cluster count/noise fraction, embedding dimensions,
  anomaly score/outlier fraction, and stability summary);
- final fit/holdout diagnostic completion.

The event stream does not invent holdout cluster labels for algorithms that lack inductive
prediction support.

### F31 — Deep Learning

`DeepLearningModel.fit(dataset, progress_callback=None)` reports one `epoch` update after
each epoch, including:

- sample-weighted mean training loss across mini-batches;
- validation loss for that epoch;
- best validation loss so far;
- current epoch, best epoch, stale-epoch count and early-stopping state.

It then emits a final `evaluation` update with the best validation loss and metrics computed
on the chronological test segment. The model's best in-memory checkpoint remains the one
selected by validation loss. The callback does not receive model weights.

## Numerical and temporal invariants

- Progress is observational. It does not change estimator parameters, random seeds, batches,
  optimizer steps, candidate ordering, model selection or numerical outputs.
- The test segment is not used to select model hyperparameters or adaptive candidates.
- The adaptive search reports validation evidence during search; test metrics are terminal only.
- Deep-learning training loss is weighted by mini-batch sample count before reporting the
  epoch mean. Validation loss is computed using the existing validation tensor.
- Progress reporting is opt-in. Existing consumers can continue calling `fit()` and
  `analyze()` with their previous arguments.
- The callback executes synchronously in the training thread. A callback/persistence exception
  propagates to the caller; the caller owns failed-run lifecycle handling.

## Tests

TD02 adds tests for:

- recorder callback-to-event conversion;
- classical-model result parity with and without telemetry;
- Gradient Boosting iteration metrics;
- adaptive candidate/generation telemetry and test-tail isolation;
- unsupervised stability events and result parity;
- Deep Learning epoch losses and result parity.

Run focused tests:

```bash
python -m pytest \
  tests/unit/test_training_observability.py \
  tests/unit/test_classical_ml.py \
  tests/unit/test_gradient_boosting.py \
  tests/unit/test_adaptive_math.py \
  tests/unit/test_unsupervised_ml.py \
  tests/unit/test_deep_learning.py -q
```

Run repository checks:

```bash
make lint
make format-check
make typecheck
make test
```

TD02 is complete when these checks pass on Python 3.11 and 3.12, along with the dedicated
F28–F31 workflows and Security scans.
