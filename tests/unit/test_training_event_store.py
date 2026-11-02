"""Tests for TD01 append-only training event persistence."""

from pathlib import Path

import pytest

from apps.dashboard.event_store import (
    CorruptTrainingEventStoreError,
    TrainingEventStore,
)
from core.contracts.training import TrainingEvent
from core.observability.training import TrainingEventRecorder


def test_event_store_round_trip_and_run_filter(tmp_path: Path) -> None:
    store = TrainingEventStore(tmp_path / "nested" / "events.jsonl")
    recorder = TrainingEventRecorder(event_sink=store.append)
    recorder.create_run(name="A", algorithm="ridge", run_id="a")
    recorder.start_run("a")
    recorder.create_run(name="B", algorithm="lstm", run_id="b")
    recorder.start_run("b")
    recorder.report_progress("b", stage="epoch", step=2, total_steps=4)

    recovered = store.read_all()
    only_b = store.read_events(run_id="b")
    after_first = store.read_events(run_id="b", after_sequence=1)

    assert len(recovered) == 5
    assert [event.run_id for event in only_b] == ["b", "b", "b"]
    assert [event.sequence for event in after_first] == [2, 3]
    assert store.read_events(run_id="missing") == ()


def test_event_store_limit_and_invalid_cursors(tmp_path: Path) -> None:
    store = TrainingEventStore(tmp_path / "events.jsonl")
    recorder = TrainingEventRecorder(event_sink=store.append)
    recorder.create_run(name="A", algorithm="ridge", run_id="a")
    recorder.start_run("a")
    recorder.warning("a", "Check source data")

    assert len(store.read_events(limit=2)) == 2
    with pytest.raises(ValueError, match="after_sequence"):
        store.read_events(after_sequence=-1)
    with pytest.raises(ValueError, match="positive"):
        store.read_events(limit=0)


def test_event_store_detects_corrupted_rows_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"broken": true}\n', encoding="utf-8")

    with pytest.raises(CorruptTrainingEventStoreError, match="line 1"):
        TrainingEventStore(path).read_all()


def test_event_store_rejects_non_event_values(tmp_path: Path) -> None:
    store = TrainingEventStore(tmp_path / "events.jsonl")
    with pytest.raises(TypeError, match="TrainingEvent"):
        store.append({"run_id": "not-an-event"})  # type: ignore[arg-type]


def test_event_store_returns_empty_when_missing(tmp_path: Path) -> None:
    assert TrainingEventStore(tmp_path / "missing.jsonl").read_all() == ()


def test_event_contract_rejects_metrics_without_values() -> None:
    with pytest.raises(ValueError, match="at least one metric"):
        TrainingEvent(
            run_id="bad-metrics",
            sequence=1,
            event_type="METRICS",
        )
