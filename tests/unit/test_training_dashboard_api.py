"""HTTP and SSE integration tests for the Training Dashboard TD03 API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from apps.dashboard.api import create_app
from apps.dashboard.event_store import TrainingEventStore
from core.contracts.training import TrainingEvent
from core.observability.training import TrainingEventRecorder


def make_recorder(
    event_sink: Callable[[TrainingEvent], None] | None = None,
) -> TrainingEventRecorder:
    recorder = TrainingEventRecorder(event_sink=event_sink)
    recorder.create_run(
        name="API integration run",
        algorithm="ridge",
        run_id="api-run-1",
        metadata={"dataset_id": "synthetic-v1"},
    )
    recorder.start_run("api-run-1", stage="training")
    recorder.report_progress(
        "api-run-1",
        stage="fit",
        step=1,
        total_steps=3,
        metrics={"train_loss": 0.42},
    )
    recorder.record_metrics("api-run-1", {"holdout_rmse": 0.18}, stage="evaluation")
    return recorder


def test_health_and_run_read_endpoints() -> None:
    recorder = make_recorder()
    client = TestClient(create_app(recorder))

    assert client.get("/healthz").json() == {"status": "ok"}

    runs = client.get("/api/v1/runs")
    assert runs.status_code == 200
    assert len(runs.json()) == 1
    assert runs.json()[0]["run_id"] == "api-run-1"
    assert runs.json()[0]["status"] == "RUNNING"

    run = client.get("/api/v1/runs/api-run-1")
    assert run.status_code == 200
    assert run.json()["metadata"]["dataset_id"] == "synthetic-v1"

    events = client.get("/api/v1/runs/api-run-1/events")
    assert events.status_code == 200
    assert [event["sequence"] for event in events.json()] == [1, 2, 3, 4]
    assert events.json()[2]["metrics"] == {"train_loss": 0.42}


def test_event_history_supports_exclusive_cursor_and_page_limit() -> None:
    client = TestClient(create_app(make_recorder()))

    response = client.get(
        "/api/v1/runs/api-run-1/events",
        params={"after_sequence": 2, "limit": 1},
    )

    assert response.status_code == 200
    assert [event["sequence"] for event in response.json()] == [3]


def test_read_endpoints_return_404_for_unknown_run() -> None:
    client = TestClient(create_app(TrainingEventRecorder()))

    assert client.get("/api/v1/runs/missing").status_code == 404
    assert client.get("/api/v1/runs/missing/events").status_code == 404
    assert client.get("/api/v1/runs/missing/events/stream").status_code == 404


def test_event_history_rejects_invalid_cursors_and_limits() -> None:
    client = TestClient(create_app(make_recorder()))

    assert (
        client.get(
            "/api/v1/runs/api-run-1/events",
            params={"after_sequence": -1},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/runs/api-run-1/events",
            params={"limit": 1_001},
        ).status_code
        == 422
    )


def test_sse_replays_events_with_sequence_ids() -> None:
    client = TestClient(create_app(make_recorder(), heartbeat_interval_seconds=0.01))

    with client.stream("GET", "/api/v1/runs/api-run-1/events/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache, no-transform"
        lines = iter(response.iter_lines())
        assert next(lines) == "retry: 3000"
        assert next(lines) == ""
        assert next(lines) == "id: 1"
        assert next(lines) == "event: run_created"
        assert next(lines).startswith('data: {"event_id":')
        assert next(lines) == ""


def test_sse_resumes_after_last_event_id_header() -> None:
    client = TestClient(create_app(make_recorder(), heartbeat_interval_seconds=0.01))

    with client.stream(
        "GET",
        "/api/v1/runs/api-run-1/events/stream?after_sequence=0",
        headers={"Last-Event-ID": "3"},
    ) as response:
        assert response.status_code == 200
        lines = iter(response.iter_lines())
        assert next(lines) == "retry: 3000"
        assert next(lines) == ""
        assert next(lines) == "id: 4"
        assert next(lines) == "event: metrics"
        assert next(lines).startswith('data: {"event_id":')


def test_sse_rejects_invalid_last_event_id() -> None:
    client = TestClient(create_app(make_recorder()))

    response = client.get(
        "/api/v1/runs/api-run-1/events/stream",
        headers={"Last-Event-ID": "not-a-sequence"},
    )

    assert response.status_code == 400


def test_sse_streams_persisted_history_after_recorder_restart(tmp_path: Path) -> None:
    store = TrainingEventStore(tmp_path / "events.jsonl")
    make_recorder(event_sink=store.append)

    # A new recorder simulates a fresh API process with only the JSONL history.
    client = TestClient(create_app(TrainingEventRecorder(), event_store=store))

    runs = client.get("/api/v1/runs")
    assert runs.status_code == 200
    assert len(runs.json()) == 1
    assert runs.json()[0]["run_id"] == "api-run-1"
    assert runs.json()[0]["status"] == "RUNNING"
    assert runs.json()[0]["event_count"] == 4
    assert runs.json()[0]["latest_metrics"] == {
        "train_loss": 0.42,
        "holdout_rmse": 0.18,
    }

    events = client.get("/api/v1/runs/api-run-1/events?after_sequence=2")
    assert [event["sequence"] for event in events.json()] == [3, 4]

    with client.stream(
        "GET",
        "/api/v1/runs/api-run-1/events/stream",
        headers={"Last-Event-ID": "3"},
    ) as response:
        lines = iter(response.iter_lines())
        assert next(lines) == "retry: 3000"
        assert next(lines) == ""
        assert next(lines) == "id: 4"
        assert next(lines) == "event: metrics"


def test_http_api_does_not_expose_run_mutation_routes() -> None:
    client = TestClient(create_app(make_recorder()))

    assert client.post("/api/v1/runs/api-run-1/cancel").status_code == 404


def test_app_rejects_non_positive_heartbeat_interval() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be positive"):
        create_app(TrainingEventRecorder(), heartbeat_interval_seconds=0)


def test_live_sse_delivers_events_published_after_subscription() -> None:
    recorder = make_recorder()
    client = TestClient(create_app(recorder, heartbeat_interval_seconds=0.05))

    with client.stream(
        "GET",
        "/api/v1/runs/api-run-1/events/stream",
        headers={"Last-Event-ID": "4"},
    ) as response:
        assert response.status_code == 200
        lines = iter(response.iter_lines())
        assert next(lines) == "retry: 3000"
        assert next(lines) == ""

        recorder.report_progress(
            "api-run-1",
            stage="validation",
            step=2,
            total_steps=3,
            metrics={"validation_loss": 0.21},
        )

        assert next(lines) == "id: 5"
        assert next(lines) == "event: progress"
        assert next(lines).startswith('data: {"event_id":')
