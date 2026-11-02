# Training Dashboard — TD03: HTTP API and Server-Sent Events

## Status and scope

TD03 adds a read-only HTTP backend around the TD01 recorder and JSONL event store.
It exposes run snapshots, ordered event history and a resumable Server-Sent Events (SSE)
stream for an individual training run.

The application is created through dependency injection. The training process and HTTP
transport must share the same `TrainingEventRecorder` instance to stream live events.
When a `TrainingEventStore` is supplied, the API can also project historical run
snapshots and replay events after the recorder has been recreated.

TD03 does not start training jobs, expose lifecycle mutation endpoints, add a graphical
interface, or change numerical model execution. Callers continue to own run creation,
start, completion, failure and cancellation.

## Components

- `apps/dashboard/api.py`: FastAPI application factory and versioned read endpoints.
- `apps/dashboard/service.py`: live/persisted query service and snapshot reconstruction.
- `apps/dashboard/event_store.py`: TD01 append-only JSONL persistence.
- `tests/unit/test_training_dashboard_api.py`: HTTP, validation, SSE and restart-recovery tests.

## Installation and application wiring

Install development and dashboard server dependencies:

```bash
python -m pip install -e ".[dev,dashboard]"
```

Create the HTTP application using the same recorder that training orchestration uses:

```python
from pathlib import Path

import uvicorn

from apps.dashboard import TrainingEventStore, create_app
from core.observability.training import TrainingEventRecorder

store = TrainingEventStore(Path("artifacts/training/events.jsonl"))
recorder = TrainingEventRecorder(event_sink=store.append)
app = create_app(recorder, event_store=store)

uvicorn.run(app, host="127.0.0.1", port=8000)
```

Bind to loopback for local development. If the dashboard is later exposed beyond the local
machine, it needs an explicit authentication, authorization, TLS and deployment policy.
TD03 does not provide those controls. CORS is not enabled; serve the frontend from the same
origin or configure a reviewed same-origin reverse proxy.

The recorder is deliberately injected. Do not create a second recorder for the training
worker: a different instance will not receive its live events. The JSONL event store is a
single-process persistence adapter in this phase; it is not a multi-process event broker.

## HTTP contract

| Method | Route | Behavior |
|---|---|---|
| GET | `/healthz` | Process liveness response |
| GET | `/api/v1/runs` | Run snapshots, newest update first |
| GET | `/api/v1/runs/{run_id}` | One run snapshot |
| GET | `/api/v1/runs/{run_id}/events` | Event history after an exclusive cursor |
| GET | `/api/v1/runs/{run_id}/events/stream` | Resumable SSE stream |

Event history accepts:

- `after_sequence`: non-negative integer, exclusive; defaults to `0`.
- `limit`: 1–1,000 events; defaults to `200`.

Unknown run IDs return HTTP 404. Invalid query values return HTTP 422. An invalid
`Last-Event-ID` header returns HTTP 400. These TD03 telemetry endpoints are read-only.
TD05 later adds a narrow, allow-listed `POST /api/v1/runs` submission endpoint and
`POST /api/v1/jobs/{run_id}/cancel`; no generic route accepts recorder state changes,
Python source, shell commands or arbitrary callables.

## SSE behavior

The stream emits standard SSE frames with:

- `id`: the event's monotonically increasing sequence number within its run;
- `event`: the lower-case event type, for example `progress` or `state_changed`;
- `data`: the complete serialized `TrainingEvent`;
- periodic comment heartbeats while a live run is idle;
- `retry: 3000` to guide client reconnection;
- `stream_end` after a historical-only replay, allowing browser clients to close the
  EventSource instead of reconnecting to a stream with no live producer.

Example browser client:

```javascript
const source = new EventSource("/api/v1/runs/run-001/events/stream");

source.addEventListener("progress", (message) => {
  const event = JSON.parse(message.data);
  console.log(event.progress_percent, event.metrics);
});

source.addEventListener("state_changed", (message) => {
  const event = JSON.parse(message.data);
  console.log(event.status);
});
```

The browser automatically sends `Last-Event-ID` when reconnecting an `EventSource`
connection. The server treats that value as the last delivered per-run sequence and emits
only later events. An explicit `after_sequence` query parameter is supported for non-browser
clients; `Last-Event-ID`, when supplied, takes precedence.

A live stream subscribes before reading the next event batch, then polls the recorder's
ordered event history after each notification. This closes the read/subscribe race and
preserves per-run ordering even when producer threads notify concurrently. A stream for a
run recovered only from JSONL replays the requested history, emits `stream_end` and closes:
that run is not active in the current process.

The transport uses the recorder's in-process listener mechanism. Deploy one API/worker
process per recorder unless a future phase introduces a cross-process event broker.

## Persistence and restart recovery

When `event_store` is passed to `create_app`:

- `GET /api/v1/runs` merges live snapshots with historical JSONL runs;
- run history can be queried after the API process has restarted;
- historical snapshots are reconstructed from each run's contiguous event sequence;
- persisted-only SSE requests replay events after their cursor and close normally.

Live in-memory snapshots take precedence over the persisted projection for the same run.
Malformed JSONL records continue to raise the TD01 corruption error rather than being
silently discarded. Event history is append-only; this phase does not prune or compact it.

Without an `event_store`, the API exposes only runs known to the injected recorder.

## Validation

Focused test command:

```bash
python -m pytest tests/unit/test_training_dashboard_api.py -q
```

Full project checks:

```bash
ruff check .
ruff format --check .
pytest -q
make typecheck
```

TD03 is considered complete when API tests, the repository CI matrix on Python 3.11/3.12,
Security and applicable upstream regression workflows pass on the final pull-request head.
