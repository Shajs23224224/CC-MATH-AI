"""HTTP integration tests for the TD05 training orchestration endpoints."""

from __future__ import annotations

from threading import Event

from fastapi.testclient import TestClient

from apps.dashboard.api import create_app
from apps.dashboard.orchestrator import TrainingOrchestrator
from core.contracts.training import TrainingProgress
from core.observability.training import TrainingEventRecorder


def test_task_catalogue_submit_jobs_queue_and_cancel_endpoints() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder, max_workers=1, max_pending=1)

    def train(parameters, progress, token):
        token.raise_if_cancelled()
        progress(
            TrainingProgress(
                stage="fit",
                progress_percent=100,
                metrics={"train_loss": float(parameters["initial_loss"])},
            )
        )

    orchestrator.register_task("classical.ridge", train, description="Ridge regression")
    client = TestClient(create_app(recorder, orchestrator=orchestrator))

    try:
        tasks = client.get("/api/v1/tasks")
        assert tasks.status_code == 200
        assert tasks.json() == [{"task_name": "classical.ridge", "description": "Ridge regression"}]

        queue = client.get("/api/v1/queue")
        assert queue.status_code == 200
        assert queue.json()["capacity"] == 2

        missing = client.post(
            "/api/v1/runs",
            json={"task_name": "unregistered", "name": "Not allowed"},
        )
        assert missing.status_code == 404

        submission = client.post(
            "/api/v1/runs",
            json={
                "task_name": "classical.ridge",
                "name": "HTTP ridge experiment",
                "parameters": {"initial_loss": 0.25},
                "metadata": {"dataset_id": "api-fixture"},
            },
        )
        assert submission.status_code == 202
        run_id = submission.json()["run_id"]
        assert submission.json()["task_name"] == "classical.ridge"

        assert orchestrator.wait(run_id, timeout=3).status == "COMPLETED"

        job = client.get(f"/api/v1/jobs/{run_id}")
        assert job.status_code == 200
        assert job.json()["status"] == "COMPLETED"

        jobs = client.get("/api/v1/jobs")
        assert jobs.status_code == 200
        assert jobs.json()[0]["run_id"] == run_id

        run = client.get(f"/api/v1/runs/{run_id}")
        assert run.status_code == 200
        assert run.json()["latest_metrics"] == {"train_loss": 0.25}

        missing_cancel = client.post("/api/v1/jobs/missing/cancel")
        assert missing_cancel.status_code == 404
    finally:
        orchestrator.shutdown()


def test_http_cancel_endpoint_stops_a_queued_job() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder, max_workers=1, max_pending=1)
    started = Event()
    release = Event()

    def blocking(_parameters, _progress, token):
        started.set()
        release.wait(timeout=3)
        token.raise_if_cancelled()

    orchestrator.register_task("blocking", blocking)
    client = TestClient(create_app(recorder, orchestrator=orchestrator))
    try:
        first = client.post(
            "/api/v1/runs",
            json={"task_name": "blocking", "name": "Active run"},
        )
        assert first.status_code == 202
        assert started.wait(timeout=2)

        second = client.post(
            "/api/v1/runs",
            json={"task_name": "blocking", "name": "Queued run"},
        )
        assert second.status_code == 202
        queued_id = second.json()["run_id"]
        assert second.json()["status"] == "QUEUED"

        cancelled = client.post(f"/api/v1/jobs/{queued_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "CANCELLED"
        assert cancelled.json()["cancellation_requested"] is True

        release.set()
        assert orchestrator.wait(first.json()["run_id"], timeout=3).status == "COMPLETED"
    finally:
        release.set()
        orchestrator.shutdown()


def test_http_submission_returns_429_when_bounded_queue_is_full() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder, max_workers=1, max_pending=0)
    started = Event()
    release = Event()

    def blocking(_parameters, _progress, token):
        started.set()
        release.wait(timeout=3)
        token.raise_if_cancelled()

    orchestrator.register_task("blocking", blocking)
    client = TestClient(create_app(recorder, orchestrator=orchestrator))
    try:
        active = client.post(
            "/api/v1/runs",
            json={"task_name": "blocking", "name": "Saturating run"},
        )
        assert active.status_code == 202
        assert started.wait(timeout=2)

        full = client.post(
            "/api/v1/runs",
            json={"task_name": "blocking", "name": "Rejected run"},
        )
        assert full.status_code == 429
        assert recorder.list_runs()[0].name == "Saturating run"
        assert orchestrator.queue_snapshot().available_slots == 0
    finally:
        release.set()
        orchestrator.shutdown()
