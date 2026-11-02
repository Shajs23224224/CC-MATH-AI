# Training Dashboard — TD05: Training orchestration

## Status and scope

TD05 adds a registry-driven orchestration layer for training tasks. It turns the TD01 run
recorder, TD02 progress callbacks, TD03 HTTP/SSE backend and TD04 browser interface into
a controllable local training workbench.

The scheduler provides:

- an explicit allow-list of registered task handlers;
- a bounded thread pool with a maximum number of running and pending jobs;
- recorder-backed QUEUED → RUNNING → COMPLETED / FAILED / CANCELLED lifecycle events;
- cooperative cancellation tokens and cancellation-aware progress callbacks;
- queue, job and task catalogue APIs;
- browser actions for submitting a registered task and requesting cancellation.

TD05 does not accept Python source, dynamic imports, shell commands or arbitrary callable
names from HTTP requests. It does not upload datasets, instantiate unrestricted models,
force-kill Python threads, persist Python return objects, or retry interrupted tasks after
a process restart.

## Architecture

```text
Browser (TD04)
  |  GET /tasks, /queue
  |  POST /runs, POST /jobs/{id}/cancel
  v
FastAPI control plane (TD03)
  v
TrainingOrchestrator (TD05)
  |-- TrainingTaskRegistry (explicit allow-list)
  |-- bounded ThreadPoolExecutor
  |-- CancellationToken
  v
Registered training adapter
  |-- resolves trusted dataset/configuration
  |-- checks cancellation at safe points
  |-- calls F28/F29/F30/F31 engine with progress callback
  v
TrainingEventRecorder (TD01/TD02)
  |-- JSONL event persistence
  |-- SSE listeners
  v
Run inspector and event timeline (TD04)
```

## Registering an approved task

Tasks are registered in application code, not by a browser request. The handler receives
three arguments:

1. `parameters`: the validated JSON object sent with the submission;
2. `progress_callback`: the existing `TrainingProgressCallback` used by F28–F31;
3. `cancellation`: a `CancellationToken` that a long-running adapter checks at safe points.

Example adapter (connect the dataset/configuration lookup to the repository's trusted
application data layer):

```python
from collections.abc import Mapping

from apps.dashboard import TrainingOrchestrator, TrainingTaskRegistry
from core.contracts.training import TrainingProgress, TrainingProgressCallback
from core.observability.training import TrainingEventRecorder


def train_classical_model(
    parameters: Mapping[str, object],
    progress: TrainingProgressCallback,
    cancellation,
):
    dataset_id = str(parameters["dataset_id"])
    dataset = dataset_repository.load(dataset_id)

    cancellation.raise_if_cancelled()
    model = build_classical_model(parameters)
    result = model.fit(dataset, progress_callback=progress)
    cancellation.raise_if_cancelled()
    return result


recorder = TrainingEventRecorder()
registry = TrainingTaskRegistry()
registry.register(
    "classical.fit",
    train_classical_model,
    description="Train an allow-listed classical ML model",
)
orchestrator = TrainingOrchestrator(
    recorder,
    registry=registry,
    max_workers=2,
    max_pending=6,
)
```

The example's `dataset_repository` and `build_classical_model` are application-owned
adapters, not public functions supplied by the scheduler. They must validate dataset IDs,
configuration values, resource limits and access permissions before calling an ML engine.

A task registry can expose distinct names for classical, deep-learning, unsupervised and
adaptive jobs. The scheduler does not guess how an arbitrary JSON parameter object maps to
an engine's typed dataset/configuration models.

## Bounded scheduling semantics

`max_workers` controls concurrent worker threads. `max_pending` controls queued jobs.
The total accepted capacity is:

```text
capacity = max_workers + max_pending
```

The default capacity is 10 (2 workers + 8 pending). A submission that would exceed capacity
is rejected immediately with HTTP 429; the scheduler does not create an unbounded queue.
Queue positions are one-based and ordered by submission time.

Terminal scheduler metadata is retained up to `max_tracked_jobs` (default 10,000), after
which oldest terminal records are pruned from the in-memory job catalogue. TD01 JSONL run
events remain the durable run history when an event store is configured.

`GET /api/v1/queue` distinguishes:
- `queued`: recorder-backed jobs waiting to start;
- `running`: jobs whose run status is RUNNING;
- `in_use`: worker-pool capacity slots still reserved, including the short finalization
  interval before a completed Future releases its slot;
- `available_slots`: capacity not currently reserved.

## Cancellation semantics

Queued jobs are cancelled immediately when the executor can remove their Future before it
starts. Running jobs receive a cancellation request; their status remains RUNNING until the
handler reaches a safe checkpoint and the worker records CANCELLED.

The supplied progress callback checks the cancellation token before recording progress.
A task handler should also call `cancellation.raise_if_cancelled()` inside long loops or
between expensive stages. This is cooperative cancellation: Python worker threads are not
forcefully terminated. A handler that never checks the token may continue until it returns.

The finalization lock serializes completion against a concurrent cancellation request. If
completion wins the lock first, a later cancellation request observes a terminal job and
does not overwrite its result.

## HTTP contract

| Method | Route | Behavior |
|---|---|---|
| GET | `/api/v1/tasks` | Registered task names and descriptions |
| GET | `/api/v1/queue` | Queue and capacity snapshot |
| POST | `/api/v1/runs` | Submit a registered training task; returns HTTP 202 |
| GET | `/api/v1/jobs` | List scheduler job snapshots |
| GET | `/api/v1/jobs/{run_id}` | Inspect scheduler metadata for one run |
| POST | `/api/v1/jobs/{run_id}/cancel` | Cancel queued job or request cooperative cancellation |

Example submission:

```json
{
  "task_name": "classical.fit",
  "name": "Ridge validation experiment",
  "parameters": {
    "dataset_id": "prices-v1",
    "seed": 31
  },
  "metadata": {
    "source": "local-research",
    "fold": 2
  }
}
```

A task must already exist in the server's registry. An unknown task returns HTTP 404; a full
queue returns HTTP 429; a shut-down scheduler returns HTTP 503. Run snapshots and event
history continue to use the TD01/TD03 `/api/v1/runs/{run_id}` and event endpoints.

## Browser controls

TD04 now includes a **New training run** dialog that loads the registered task catalogue and
submits task parameters/metadata as JSON. The run inspector displays a cancellation control
for scheduler-managed QUEUED/RUNNING jobs. It distinguishes an immediate queued cancellation
from a running cancellation request.

If the task registry is empty, the dialog explains that approved handlers must be registered
in application code before jobs can be submitted.

## Application wiring

Build the recorder, store, registry and scheduler once, then inject that same recorder and
scheduler into the FastAPI application:

```python
from pathlib import Path

import uvicorn

from apps.dashboard import (
    TrainingEventStore,
    TrainingOrchestrator,
    TrainingTaskRegistry,
    create_app,
)
from core.observability.training import TrainingEventRecorder

store = TrainingEventStore(Path("artifacts/training/events.jsonl"))
recorder = TrainingEventRecorder(event_sink=store.append)
registry = TrainingTaskRegistry()
registry.register("classical.fit", train_classical_model)
orchestrator = TrainingOrchestrator(
    recorder,
    registry=registry,
    max_workers=2,
    max_pending=6,
)
app = create_app(recorder, event_store=store, orchestrator=orchestrator)

uvicorn.run(app, host="127.0.0.1", port=8000)
```

The application owns shutdown of an internally created scheduler. When an orchestrator is
injected, its owner is responsible for calling `orchestrator.shutdown()` after the app stops.

## Persistence and process boundaries

The TD01 event store persists run events, including submission, start, progress and terminal
state transitions. TD05's live Future objects, task handlers, queue positions and cancellation
tokens are process-local. After a server restart, historical run snapshots remain available
from JSONL, but unfinished jobs are not automatically restarted or requeued. Operators must
inspect the run state and explicitly submit a new job if appropriate.

The recorder, event store and worker pool are single-process components in this phase. A
separate worker process requires a future durable queue and cross-process event broker.

## Security and resource controls

- Bind to `127.0.0.1` for local development.
- Do not expose the control API publicly without authentication, authorization, TLS, rate
  limiting, job quotas and a reviewed deployment policy.
- Only code-registered handlers can be invoked; HTTP parameters are data, never code.
- Handlers must validate input sizes, dataset access and model resource budgets.
- API error messages are generic. Full exception details remain in server logs.
- Cancellation is cooperative and cannot safely kill arbitrary Python code.

## Validation

Focused tests:

```bash
python -m pytest \
  tests/unit/test_training_orchestrator.py \
  tests/unit/test_training_orchestration_api.py \
  tests/unit/test_training_dashboard_api.py \
  tests/unit/test_training_dashboard_ui.py -q
```

Full checks:

```bash
ruff check .
ruff format --check .
make typecheck
pytest -q
```

TD05 completion requires the CI matrix on Python 3.11/3.12, Security, and relevant F28–F31
regression workflows to pass on the final pull-request head.
