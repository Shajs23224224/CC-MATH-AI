"""Query service joining live recorder state with persisted TD01 history."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from apps.dashboard.event_store import TrainingEventStore
from core.contracts.training import TrainingEvent, TrainingRunSnapshot
from core.observability.training import (
    TrainingEventRecorder,
    TrainingRunNotFoundError,
)

_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
EventListener = Callable[[TrainingEvent], None]


class TrainingEventHistoryError(RuntimeError):
    """Raised when persisted events cannot form a coherent run history."""


class TrainingDashboardService:
    """Read live run state and recover historical snapshots from the JSONL store."""

    def __init__(
        self,
        recorder: TrainingEventRecorder,
        *,
        event_store: TrainingEventStore | None = None,
    ) -> None:
        self.recorder = recorder
        self.event_store = event_store

    def list_runs(self) -> tuple[TrainingRunSnapshot, ...]:
        """Return live and persisted run snapshots without duplicate run IDs."""
        snapshots = {item.run_id: item for item in self.recorder.list_runs()}
        if self.event_store is not None:
            grouped: dict[str, list[TrainingEvent]] = defaultdict(list)
            for event in self.event_store.read_all():
                grouped[event.run_id].append(event)
            for run_id, events in grouped.items():
                if run_id not in snapshots:
                    snapshots[run_id] = self._snapshot_from_events(tuple(events))
        return tuple(
            sorted(snapshots.values(), key=lambda item: item.updated_at, reverse=True)
        )

    def snapshot(self, run_id: str) -> TrainingRunSnapshot:
        """Return live state when available, otherwise project persisted history."""
        try:
            return self.recorder.snapshot(run_id)
        except TrainingRunNotFoundError:
            events = self._stored_events(run_id)
            if not events:
                raise
            return self._snapshot_from_events(events)

    def events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> tuple[TrainingEvent, ...]:
        """Return an exclusive per-run event cursor from memory or durable storage."""
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        try:
            self.recorder.snapshot(run_id)
        except TrainingRunNotFoundError:
            events = self._stored_events(run_id)
            if not events:
                raise
            filtered = tuple(event for event in events if event.sequence > after_sequence)
        else:
            filtered = self.recorder.events(run_id, after_sequence=after_sequence)
        return filtered if limit is None else filtered[:limit]

    def is_live_run(self, run_id: str) -> bool:
        """Return whether this process currently owns the run's recorder state."""
        try:
            self.recorder.snapshot(run_id)
        except TrainingRunNotFoundError:
            return False
        return True

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """Delegate an event listener subscription to the live recorder."""
        return self.recorder.subscribe(listener)

    def _stored_events(self, run_id: str) -> tuple[TrainingEvent, ...]:
        if self.event_store is None:
            return ()
        return self.event_store.read_events(run_id=run_id)

    @staticmethod
    def _snapshot_from_events(
        events: tuple[TrainingEvent, ...],
    ) -> TrainingRunSnapshot:
        """Rebuild the immutable latest run snapshot from its append-only event log."""
        if not events:
            raise TrainingEventHistoryError("cannot project an empty training event history")
        ordered = tuple(sorted(events, key=lambda item: item.sequence))
        first = ordered[0]
        if first.sequence != 1 or first.event_type != "RUN_CREATED":
            raise TrainingEventHistoryError("training event history must begin with RUN_CREATED")

        snapshot = TrainingRunSnapshot(
            run_id=first.run_id,
            name=first.run_name or "",
            algorithm=first.algorithm or "",
            status="QUEUED",
            created_at=first.occurred_at,
            updated_at=first.occurred_at,
            event_count=1,
            message=first.message,
            metadata=dict(first.run_metadata),
        )
        for expected_sequence, event in enumerate(ordered[1:], start=2):
            if event.run_id != first.run_id or event.sequence != expected_sequence:
                raise TrainingEventHistoryError(
                    f"non-contiguous event history for run {first.run_id}"
                )
            updates: dict[str, object] = {
                "updated_at": event.occurred_at,
                "event_count": event.sequence,
            }
            if event.status is not None:
                updates["status"] = event.status
                if event.status == "RUNNING":
                    updates["started_at"] = event.occurred_at
                if event.status in _TERMINAL_STATUSES:
                    updates["ended_at"] = event.occurred_at
            if event.stage is not None:
                updates["current_stage"] = event.stage
            if event.progress_percent is not None:
                updates["progress_percent"] = event.progress_percent
            if event.step is not None:
                updates["current_step"] = event.step
                updates["total_steps"] = event.total_steps
            if event.metrics:
                updates["latest_metrics"] = {
                    **snapshot.latest_metrics,
                    **event.metrics,
                }
            if event.error_code is not None:
                updates["error_code"] = event.error_code
            if event.message is not None:
                updates["message"] = event.message
            snapshot = snapshot.model_copy(update=updates)
        return snapshot


__all__ = ["TrainingDashboardService", "TrainingEventHistoryError"]
