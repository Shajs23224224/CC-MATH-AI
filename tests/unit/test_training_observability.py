"""Tests for TD01 training events, recorder state and observer isolation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from core.contracts.training import TrainingProgress
from core.observability.training import (
    TrainingEventRecorder,
    TrainingRunNotFoundError,
    TrainingStateError,
)

def test_recorder_tracks_ordered_run_lifecycle_and_metrics() -> None:
    recorder = TrainingEventRecorder()
    created = recorder.create_run(
        name="LSTM experiment",
        algorithm="lstm",
        run_id="run-lstm-001",
        metadata={"dataset_id": "prices-v1", "seed": 31},
    )

    assert created.status == "QUEUED"
    assert created.event_count == 1
    assert created.metadata["dataset_id"] == "prices-v1"
    created_event = recorder.events("run-lstm-001")[0]
    assert created_event.run_name == "LSTM experiment"
    assert created_event.algorithm == "lstm"
    assert created_event.run_metadata["seed"] == 31

    recorder.start_run("run-lstm-001", stage="training")
    progress = recorder.report_progress(
        "run-lstm-001",
        stage="epoch",
        step=3,
        total_steps=10,
        metrics={"train_loss": 0.42, "validation_loss": 0.51},
    )
    recorder.record_metrics(
        "run-lstm-001",
        {"mae": 0.12, "rmse": 0.19},
        stage="evaluation",
    )
    recorder.complete("run-lstm-001")

    snapshot = recorder.snapshot("run-lstm-001")
    events = recorder.events("run-lstm-001")

    assert progress.progress_percent == 30.0
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
    assert snapshot.status == "COMPLETED"
    assert snapshot.progress_percent == 100.0
    assert snapshot.current_stage == "evaluation"
    assert snapshot.latest_metrics == {
        "train_loss": 0.42,
        "validation_loss": 0.51,
        "mae": 0.12,
        "rmse": 0.19,
    }
    assert snapshot.started_at is not None
    assert snapshot.ended_at is not None
    assert len(recorder.list_runs()) == 1

def test_recorder_keeps_runs_and_event_cursors_separate() -> None:
    received: list[str] = []
    recorder = TrainingEventRecorder()
    recorder.subscribe(lambda event: received.append(event.run_id))
    recorder.create_run(name="First", algorithm="ridge", run_id="run-1")
    recorder.create_run(name="Second", algorithm="random_forest", run_id="run-2")
    recorder.start_run("run-1")
    recorder.start_run("run-2")
    recorder.report_progress("run-1", stage="fit", progress_percent=50.0)

    assert len(recorder.events("run-1")) == 3
    assert [event.sequence for event in recorder.events("run-2")] == [1, 2]
    assert recorder.events("run-1", after_sequence=2)[0].sequence == 3
    assert set(received) == {"run-1", "run-2"}

def test_recorder_rejects_invalid_lifecycle_operations() -> None:
    recorder = TrainingEventRecorder()
    recorder.create_run(name="Queued", algorithm="ridge", run_id="queued")

    with pytest.raises(TrainingStateError, match="QUEUED"):
        recorder.report_progress("queued", stage="fit", progress_percent=10.0)
    with pytest.raises(TrainingStateError, match="already exists"):
        recorder.create_run(name="Duplicate", algorithm="ridge", run_id="queued")

    recorder.start_run("queued")
    recorder.complete("queued")
    with pytest.raises(TrainingStateError, match="COMPLETED"):
        recorder.cancel("queued")
    with pytest.raises(TrainingRunNotFoundError):
        recorder.snapshot("missing")

def test_recorder_failure_and_cancellation_are_terminal() -> None:
    recorder = TrainingEventRecorder()
    recorder.create_run(name="Failure", algorithm="lstm", run_id="failure")
    recorder.start_run("failure")
    failed = recorder.fail(
        "failure",
        error_code="NUMERIC_DIVERGENCE",
        message="Training loss became non-finite",
        stage="epoch",
    )

    assert failed.status == "FAILED"
    assert recorder.snapshot("failure").error_code == "NUMERIC_DIVERGENCE"
    with pytest.raises(TrainingStateError):
        recorder.complete("failure")

    recorder.create_run(name="Cancelled", algorithm="ridge", run_id="cancelled")
    recorder.cancel("cancelled")
    assert recorder.snapshot("cancelled").status == "CANCELLED"

def test_listener_failure_does_not_abort_recording() -> None:
    recorder = TrainingEventRecorder()

    def broken_listener(_event: object) -> None:
        raise RuntimeError("observer failed")

    recorder.subscribe(broken_listener)
    recorder.create_run(name="Observed", algorithm="ridge", run_id="observed")
    recorder.start_run("observed")

    assert recorder.snapshot("observed").status == "RUNNING"

def test_event_sink_failure_does_not_advance_run_state() -> None:
    def broken_sink(_event: object) -> None:
        raise OSError("disk unavailable")

    recorder = TrainingEventRecorder(event_sink=broken_sink)
    with pytest.raises(OSError, match="disk unavailable"):
        recorder.create_run(name="Not saved", algorithm="ridge", run_id="not-saved")
    with pytest.raises(TrainingRunNotFoundError):
        recorder.snapshot("not-saved")

def test_event_contract_rejects_inconsistent_progress_and_naive_timestamp() -> None:
    from core.contracts.training import TrainingEvent

    with pytest.raises(ValueError, match="step and total_steps"):
        TrainingEvent(
            run_id="bad-progress",
            sequence=1,
            event_type="PROGRESS",
            step=1,
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        TrainingEvent(
            run_id="bad-time",
            sequence=1,
            event_type="WARNING",
            message="naive",
            occurred_at=datetime(2026, 1, 1),
        )

def test_event_sink_receives_a_contiguous_sequence_per_run() -> None:
    persisted = []
    recorder = TrainingEventRecorder(event_sink=persisted.append)
    recorder.create_run(name="A", algorithm="ridge", run_id="a")
    recorder.start_run("a")
    recorder.warning("a", "Minor input quality issue")
    recorder.complete("a")

    assert [event.sequence for event in persisted] == [1, 2, 3, 4]

def test_concurrent_metric_events_keep_contiguous_sequences() -> None:
    recorder = TrainingEventRecorder()
    recorder.create_run(name="Concurrent", algorithm="ridge", run_id="concurrent")
    recorder.start_run("concurrent")

    def record(index: int) -> None:
        recorder.record_metrics("concurrent", {f"metric_{index}": float(index)})

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(record, range(40)))

    events = recorder.events("concurrent")
    snapshot = recorder.snapshot("concurrent")
    assert [event.sequence for event in events] == list(range(1, 43))
    assert len(snapshot.latest_metrics) == 40
    assert snapshot.event_count == 42

def test_recorder_progress_callback_bridges_engine_metrics() -> None:
    recorder = TrainingEventRecorder()
    recorder.create_run(name="Observed model", algorithm="ridge", run_id="callback-run")
    recorder.start_run("callback-run", stage="training")

    callback = recorder.progress_callback("callback-run")
    callback(
        TrainingProgress(
            stage="validation",
            step=3,
            total_steps=10,
            progress_percent=42.0,
            metrics={"validation_rmse": 0.25},
            attributes={"best_iteration": 3, "test_used_for_selection": False},
            message="Validation iteration completed",
        )
    )

    event = recorder.events("callback-run")[-1]
    snapshot = recorder.snapshot("callback-run")
    assert event.stage == "validation"
    assert event.progress_percent == 42.0
    assert event.metrics["validation_rmse"] == 0.25
    assert event.attributes["best_iteration"] == 3
    assert snapshot.latest_metrics["validation_rmse"] == 0.25
