"""Bounded scheduler, lifecycle and cancellation tests for TD05."""

from __future__ import annotations

from threading import Event

import pytest

from core.contracts.training import TrainingProgress, TrainingTaskSubmission
from core.observability.training import TrainingEventRecorder
from apps.dashboard.orchestrator import (
    TrainingOrchestrator,
    TrainingQueueFullError,
    TrainingTaskNotFoundError,
    TrainingTaskRegistry,
)


def test_registry_lists_only_explicitly_registered_tasks() -> None:
    registry = TrainingTaskRegistry()

    def runner(_parameters: object, _progress: object, _token: object) -> None:
        return None

    registry.register("classical.ridge", runner, description="Ridge regression")
    assert [item.task_name for item in registry.list_tasks()] == ["classical.ridge"]
    assert registry.list_tasks()[0].description == "Ridge regression"
    assert registry.unregister("classical.ridge") is True
    assert registry.list_tasks() == ()
    assert registry.unregister("classical.ridge") is False


@pytest.mark.parametrize("task_name", ["", "Uppercase", "../unsafe", "not a name", "a" * 81])
def test_registry_rejects_invalid_task_names(task_name: str) -> None:
    registry = TrainingTaskRegistry()
    with pytest.raises(ValueError, match="task_name"):
        registry.register(task_name, lambda *_args: None)


def test_registry_rejects_duplicate_task_names() -> None:
    registry = TrainingTaskRegistry()
    registry.register("ridge", lambda *_args: None)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("ridge", lambda *_args: None)


def test_submit_records_lifecycle_progress_and_metrics() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder, max_workers=1, max_pending=1)

    def train(parameters, progress, token):
        token.raise_if_cancelled()
        assert parameters["epochs"] == 3
        progress(
            TrainingProgress(
                stage="epoch",
                step=3,
                total_steps=3,
                metrics={"train_loss": 0.125},
            )
        )
        return {"ignored": "result"}

    orchestrator.register_task("classical.ridge", train, description="Ridge model")
    try:
        submitted = orchestrator.submit(
            "classical.ridge",
            name="Ridge smoke test",
            parameters={"epochs": 3},
            metadata={"dataset_id": "synthetic-v1"},
        )
        assert submitted.status in {"QUEUED", "RUNNING"}

        finished = orchestrator.wait(submitted.run_id, timeout=3)
        events = recorder.events(submitted.run_id)

        assert finished.status == "COMPLETED"
        assert finished.started_at is not None
        assert finished.finished_at is not None
        assert finished.cancellation_requested is False
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert [event.event_type for event in events] == [
            "RUN_CREATED",
            "STATE_CHANGED",
            "PROGRESS",
            "STATE_CHANGED",
        ]
        assert recorder.snapshot(submitted.run_id).latest_metrics == {"train_loss": 0.125}
        assert orchestrator.queue_snapshot().running == 0
    finally:
        orchestrator.shutdown()


def test_scheduler_bounds_running_and_pending_jobs() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder, max_workers=1, max_pending=1)
    started = Event()
    release = Event()

    def blocking_task(_parameters, _progress, token):
        started.set()
        release.wait(timeout=3)
        token.raise_if_cancelled()

    orchestrator.register_task("blocking", blocking_task)
    try:
        first = orchestrator.submit("blocking", name="First")
        assert started.wait(timeout=2)
        second = orchestrator.submit("blocking", name="Second")
        assert second.status == "QUEUED"
        assert second.queue_position == 1

        queue = orchestrator.queue_snapshot()
        assert queue.max_workers == 1
        assert queue.max_pending == 1
        assert queue.capacity == 2
        assert queue.running == 1
        assert queue.queued == 1
        assert queue.available_slots == 0

        with pytest.raises(TrainingQueueFullError):
            orchestrator.submit("blocking", name="Third")

        cancelled = orchestrator.cancel(second.run_id)
        assert cancelled.status == "CANCELLED"
        assert recorder.snapshot(second.run_id).status == "CANCELLED"

        release.set()
        assert orchestrator.wait(first.run_id, timeout=3).status == "COMPLETED"
    finally:
        release.set()
        orchestrator.shutdown()


def test_running_job_cancels_at_next_progress_checkpoint() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder, max_workers=1, max_pending=0)
    started = Event()
    release = Event()

    def cooperative_task(_parameters, progress, _token):
        started.set()
        release.wait(timeout=3)
        # The orchestrator's progress adapter checks the token before recording.
        progress(TrainingProgress(stage="epoch", step=1, total_steps=2))

    orchestrator.register_task("cooperative", cooperative_task)
    try:
        submitted = orchestrator.submit("cooperative", name="Cooperative cancellation")
        assert started.wait(timeout=2)

        requested = orchestrator.cancel(submitted.run_id)
        assert requested.cancellation_requested is True
        assert requested.status == "RUNNING"

        release.set()
        finished = orchestrator.wait(submitted.run_id, timeout=3)
        assert finished.status == "CANCELLED"
        assert recorder.snapshot(submitted.run_id).status == "CANCELLED"
        assert [event.event_type for event in recorder.events(submitted.run_id)][-1] == "STATE_CHANGED"
    finally:
        release.set()
        orchestrator.shutdown()


def test_task_failure_is_recorded_without_exposing_exception_details() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder, max_workers=1, max_pending=0)

    def broken_task(_parameters, _progress, _token):
        raise RuntimeError("private path /tmp/secret-dataset.csv")

    orchestrator.register_task("broken", broken_task)
    try:
        submitted = orchestrator.submit("broken", name="Failure case")
        finished = orchestrator.wait(submitted.run_id, timeout=3)
        snapshot = recorder.snapshot(submitted.run_id)
        error_event = recorder.events(submitted.run_id)[-1]

        assert finished.status == "FAILED"
        assert snapshot.error_code == "TASK_EXECUTION_FAILED"
        assert "private path" not in (snapshot.message or "")
        assert error_event.event_type == "ERROR"
        assert error_event.message == "Training task failed; inspect server logs"
    finally:
        orchestrator.shutdown()


def test_unknown_task_is_rejected_before_a_run_is_created() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder)

    try:
        with pytest.raises(TrainingTaskNotFoundError):
            orchestrator.submit("not.registered", name="No handler")
        assert recorder.list_runs() == ()
    finally:
        orchestrator.shutdown()


def test_shutdown_rejects_new_submissions() -> None:
    recorder = TrainingEventRecorder()
    orchestrator = TrainingOrchestrator(recorder)
    orchestrator.register_task("known", lambda *_args: None)
    orchestrator.shutdown()

    from apps.dashboard.orchestrator import TrainingOrchestratorClosedError

    with pytest.raises(TrainingOrchestratorClosedError):
        orchestrator.submit("known", name="After shutdown")


def test_task_submission_rejects_oversized_and_non_finite_parameters() -> None:
    with pytest.raises(ValueError, match="64 KiB"):
        TrainingTaskSubmission(
            task_name="classical.fit",
            name="Large input",
            parameters={"payload": "x" * (65 * 1024)},
        )

    with pytest.raises(ValueError, match="finite JSON"):
        TrainingTaskSubmission(
            task_name="classical.fit",
            name="Non-finite input",
            parameters={"threshold": float("nan")},
        )


def test_task_submission_bounds_metadata_and_parameter_count() -> None:
    with pytest.raises(ValueError, match="metadata string values"):
        TrainingTaskSubmission(
            task_name="classical.fit",
            name="Large metadata",
            metadata={"description": "x" * 513},
        )

    with pytest.raises(ValueError, match="64 task parameters"):
        TrainingTaskSubmission(
            task_name="classical.fit",
            name="Too many parameters",
            parameters={f"parameter_{index}": index for index in range(65)},
        )
