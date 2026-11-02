# Training Dashboard — TD04: Web interface

## Status and scope

TD04 adds a responsive, same-origin browser interface for the Training Dashboard. The UI is
served by the existing FastAPI application and consumes the TD03 read-only API and SSE
stream. It does not introduce a separate frontend server, JavaScript build chain, npm
runtime, charting library or browser-to-model execution path.

## Frontend files

- `apps/dashboard/static/index.html`: accessible page structure and dashboard regions.
- `apps/dashboard/static/styles.css`: responsive dark interface, state indicators and
  reduced-motion support.
- `apps/dashboard/static/app.js`: REST data loading, search/filter state, run inspection,
  event timeline and EventSource reconnection.
- `apps/dashboard/api.py`: serves the frontend at `/dashboard/` and redirects `/` there.
- `pyproject.toml`: includes static assets in built Python packages.
- `tests/unit/test_training_dashboard_ui.py`: route, asset and same-origin delivery tests.

## Interface capabilities

### Training overview

The summary cards show:

- total recorded runs;
- currently running runs;
- completed runs;
- failed and cancelled runs.

The run table displays name, algorithm, state, latest progress, current stage and relative
update time. Search matches run name, algorithm and run ID. Status filtering supports
QUEUED, RUNNING, COMPLETED, FAILED and CANCELLED. Refresh reloads data from the API, and a
30-second refresh keeps the list current even when no event stream is selected.

### Run inspector

Selecting a run loads its current snapshot and recent event history. The inspector shows
status, algorithm, progress, stage, step counters, creation time, duration, event count,
latest numeric metrics and a chronological event timeline.

The browser builds dynamic content with DOM text nodes rather than inserting API-provided
strings as HTML. This prevents run names, metric keys and event messages from being treated
as executable markup.

### Live events

For a RUNNING run, the UI opens:

```text
GET /api/v1/runs/{run_id}/events/stream
```

It listens for the TD01 event types `run_created`, `state_changed`, `progress`,
`metrics`, `warning` and `error`. Incoming events update the selected run, summary,
progress display, metrics and activity timeline without a full-page reload.

The browser uses the SSE event ID as its reconnect cursor. Native `EventSource` reconnection
therefore resumes from `Last-Event-ID` as implemented by TD03. The stream is closed when
the selected run becomes terminal or when the user selects another run. A persisted-only
run sends TD03's `stream_end` marker after replay, so the browser does not reconnect
indefinitely to historical-only data. The inspector requests the most recent 200 events and
retains a bounded in-memory activity window while the stream remains open.

## Run locally

Install the server dependencies:

```bash
python -m pip install -e ".[dashboard]"
```

Create the FastAPI app with the recorder shared by the training process and run Uvicorn:

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

Open `http://127.0.0.1:8000/dashboard/`. The API root `/` redirects to the dashboard.
The dashboard and API share the same origin; no CORS configuration is needed.

This example serves persisted run history and any training runs created through the same
recorder instance. A separate training process needs a future cross-process event broker to
publish live notifications; TD04 does not add one.

For convenience in an application entry point, the API factory is still injectable. Do not
create a disconnected recorder and expect it to observe another process's in-memory events.

## Security and operating boundaries

- The dashboard is read-only. It does not create, cancel, restart or modify training runs.
- Bind to `127.0.0.1` for local development.
- Do not expose it publicly without authentication, authorization, TLS, rate controls and a
  reviewed deployment policy.
- The API returns event metadata and metrics already emitted by the training engine; do not
  put secrets, credentials or private data in run metadata or event messages.
- Static assets are local and contain no third-party CDN dependency.
- The event stream uses the TD01 in-process recorder listener and TD03 JSONL history path.
  Cross-process live delivery remains out of scope.

## Validation

Focused test command:

```bash
python -m pytest \
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

TD04 completion requires the general CI matrix on Python 3.11/3.12, Security and applicable
F28–F31 regression workflows to pass on the final pull-request head.
