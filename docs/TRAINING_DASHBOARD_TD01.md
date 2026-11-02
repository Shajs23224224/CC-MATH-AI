# Training Dashboard — TD01: Event and observability foundation

## Status

TD01 implements the first reusable observability layer for the Training Dashboard.
It does not add a graphical interface, web server, training command or instrumentation
inside the existing model engines. Those integrations belong to later phases.

## Components

- `core/contracts/training.py`: immutable Pydantic contracts for training events and run snapshots.
- `core/observability/training.py`: thread-safe per-run event recorder, lifecycle validation,
  sequence numbering and listener notifications.
- `apps/dashboard/event_store.py`: append-only JSONL persistence and validated event recovery.
- `tests/unit/test_training_observability.py` and
  `tests/unit/test_training_event_store.py`: contract, lifecycle, isolation and persistence tests.

## Event model

Each event includes a unique event ID, run ID, per-run sequence number, UTC timestamp,
schema version and event type. Optional fields carry lifecycle status, stage, progress,
step counters, numeric metrics, scalar attributes and error information.

Supported event types are:

- `RUN_CREATED`
- `STATE_CHANGED`
- `PROGRESS`
- `METRICS`
- `WARNING`
- `ERROR`

Run lifecycle:

```text
QUEUED -> RUNNING -> COMPLETED
                   -> FAILED
                   -> CANCELLED

QUEUED -> FAILED
QUEUED -> CANCELLED
```

Progress and metrics are accepted only while a run is RUNNING. Warnings are allowed
while QUEUED or RUNNING. Terminal states cannot transition again. Sequence numbers start
at one and increase independently for every run.

## Example

```python
from apps.dashboard.event_store import TrainingEventStore
from core.observability.training import TrainingEventRecorder

store = TrainingEventStore("artifacts/training/events.jsonl")
recorder = TrainingEventRecorder(event_sink=store.append)

recorder.create_run(
    name="LSTM experiment",
    algorithm="lstm",
    run_id="experiment-001",
    metadata={"dataset_id": "prices-v1", "seed": 31},
)
recorder.start_run("experiment-001", stage="training")
recorder.report_progress(
    "experiment-001",
    stage="epoch",
    step=12,
    total_steps=50,
    metrics={"train_loss": 0.18, "validation_loss": 0.24},
)
recorder.complete("experiment-001")

events = store.read_events(run_id="experiment-001")
```

## Reliability and boundaries

- Events are validated before they are accepted.
- The optional persistence sink is called before in-memory state advances. A persistence
  error is propagated to the caller.
- Listener exceptions are isolated so visualization code cannot interrupt model execution.
- The recorder is thread-safe and assigns sequence numbers under a lock.
- The JSONL store protects concurrent threads within one process. Multi-process writes are
  not supported in TD01.
- A malformed persisted line fails with its line number; the store does not silently skip it.
- The recorder and event store have no dependency on Streamlit, a browser, a web framework,
  PyTorch or scikit-learn.
- Metrics are observations, not model-selection instructions. The recorder does not access
  training, validation or test data and cannot alter model behavior.

## Validation

Run the focused tests:

```bash
python -m pytest tests/unit/test_training_observability.py tests/unit/test_training_event_store.py -q
```

Run repository checks:

```bash
make test
make lint
make format-check
```

TD01 is complete when the new tests pass on supported Python versions, static checks pass,
and the existing quantitative and machine-learning regression suites remain unchanged.
