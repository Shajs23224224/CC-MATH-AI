"""Thread-safe event recorder for independent training runs (TD01)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Final
from uuid import uuid4

from core.contracts.training import (
    TrainingEvent,
    TrainingEventType,
    TrainingProgress,
    TrainingProgressCallback,
    TrainingRunSnapshot,
    TrainingScalar,
    TrainingStatus,
)

EventSink = Callable[[TrainingEvent], None]
EventListener = Callable[[TrainingEvent], None]

_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
_ACTIVE_STATUSES: Final[frozenset[str]] = frozenset({"QUEUED", "RUNNING"})


class TrainingRunNotFoundError(KeyError):
    """Raised when an event references a run not registered by this recorder."""


class TrainingStateError(RuntimeError):
    """Raised when a training run receives an invalid lifecycle operation."""


@dataclass(slots=True)
class _RunState:
    snapshot: TrainingRunSnapshot
    events: list[TrainingEvent] = field(default_factory=list)


class TrainingEventRecorder:
    """Record ordered run events, expose snapshots and notify lightweight listeners.

    An optional event_sink is called before in-memory state is changed. A persistence
    failure therefore propagates to the caller without silently advancing the run.
    Listener failures are isolated so a UI observer cannot abort model training.
    """

    def __init__(self, event_sink: EventSink | None = None) -> None:
        self._event_sink = event_sink
        self._runs: dict[str, _RunState] = {}
        self._listeners: dict[int, EventListener] = {}
        self._next_listener_id = 1
        self._lock = RLock()

    def create_run(
        self,
        *,
        name: str,
        algorithm: str,
        run_id: str | None = None,
        metadata: Mapping[str, TrainingScalar] | None = None,
    ) -> TrainingRunSnapshot:
        """Register a new QUEUED run and emit its first event."""
        resolved_run_id = run_id or str(uuid4())
        now = datetime.now(UTC)
        snapshot = TrainingRunSnapshot(
            run_id=resolved_run_id,
            name=name,
            algorithm=algorithm,
            status="QUEUED",
            created_at=now,
            updated_at=now,
            event_count=1,
            metadata=dict(metadata or {}),
        )
        event = TrainingEvent(
            run_id=resolved_run_id,
            sequence=1,
            occurred_at=now,
            event_type="RUN_CREATED",
            status="QUEUED",
            run_name=name,
            algorithm=algorithm,
            run_metadata=dict(metadata or {}),
            message="Training run created",
        )
        with self._lock:
            if resolved_run_id in self._runs:
                raise TrainingStateError(f"training run already exists: {resolved_run_id}")
            self._persist(event)
            self._runs[resolved_run_id] = _RunState(snapshot=snapshot, events=[event])
            listeners = tuple(self._listeners.values())
        self._notify(event, listeners)
        return snapshot

    def start_run(self, run_id: str, *, stage: str = "initialization") -> TrainingEvent:
        """Move a queued run into RUNNING."""
        return self._emit(
            run_id,
            event_type="STATE_CHANGED",
            expected_statuses=frozenset({"QUEUED"}),
            status="RUNNING",
            stage=stage,
            message="Training started",
        )

    def report_progress(
        self,
        run_id: str,
        *,
        stage: str,
        progress_percent: float | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        metrics: Mapping[str, float] | None = None,
        attributes: Mapping[str, TrainingScalar] | None = None,
        message: str | None = None,
    ) -> TrainingEvent:
        """Report stage progress and optional metrics from the active computation."""
        if progress_percent is None and step is not None and total_steps is not None:
            progress_percent = 100.0 * step / total_steps
        return self._emit(
            run_id,
            event_type="PROGRESS",
            expected_statuses=frozenset({"RUNNING"}),
            stage=stage,
            progress_percent=progress_percent,
            step=step,
            total_steps=total_steps,
            metrics=dict(metrics or {}),
            attributes=dict(attributes or {}),
            message=message,
        )

    def progress_callback(self, run_id: str) -> TrainingProgressCallback:
        """Return a callback that forwards algorithm progress into this recorder.

        The run must already exist and be RUNNING before the callback is invoked.
        """
        def report(progress: TrainingProgress) -> None:
            self.report_progress(
                run_id,
                stage=progress.stage,
                progress_percent=progress.progress_percent,
                step=progress.step,
                total_steps=progress.total_steps,
                metrics=progress.metrics,
                attributes=progress.attributes,
                message=progress.message,
            )

        return report

    def record_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, float],
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> TrainingEvent:
        """Record a named group of metrics without implying epoch-based progress."""
        return self._emit(
            run_id,
            event_type="METRICS",
            expected_statuses=frozenset({"RUNNING"}),
            stage=stage,
            metrics=dict(metrics),
            message=message,
        )

    def warning(
        self,
        run_id: str,
        message: str,
        *,
        stage: str | None = None,
        attributes: Mapping[str, TrainingScalar] | None = None,
    ) -> TrainingEvent:
        """Record a non-terminal warning."""
        return self._emit(
            run_id,
            event_type="WARNING",
            expected_statuses=_ACTIVE_STATUSES,
            stage=stage,
            message=message,
            attributes=dict(attributes or {}),
        )

    def complete(self, run_id: str, *, message: str = "Training completed") -> TrainingEvent:
        """Finish an active run successfully."""
        return self._emit(
            run_id,
            event_type="STATE_CHANGED",
            expected_statuses=frozenset({"RUNNING"}),
            status="COMPLETED",
            progress_percent=100.0,
            message=message,
        )

    def fail(
        self,
        run_id: str,
        *,
        error_code: str,
        message: str,
        stage: str | None = None,
    ) -> TrainingEvent:
        """Mark a queued or running run as failed."""
        return self._emit(
            run_id,
            event_type="ERROR",
            expected_statuses=_ACTIVE_STATUSES,
            status="FAILED",
            stage=stage,
            error_code=error_code,
            message=message,
        )

    def cancel(self, run_id: str, *, message: str = "Training cancelled") -> TrainingEvent:
        """Cancel a queued or running run."""
        return self._emit(
            run_id,
            event_type="STATE_CHANGED",
            expected_statuses=_ACTIVE_STATUSES,
            status="CANCELLED",
            message=message,
        )

    def snapshot(self, run_id: str) -> TrainingRunSnapshot:
        """Return the latest immutable snapshot for a run."""
        with self._lock:
            return self._get_run(run_id).snapshot

    def list_runs(self) -> tuple[TrainingRunSnapshot, ...]:
        """Return snapshots ordered by most recently updated."""
        with self._lock:
            snapshots = (state.snapshot for state in self._runs.values())
            return tuple(sorted(snapshots, key=lambda item: item.updated_at, reverse=True))

    def events(self, run_id: str, *, after_sequence: int = 0) -> tuple[TrainingEvent, ...]:
        """Return this recorder's events for one run after the supplied sequence."""
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        with self._lock:
            run = self._get_run(run_id)
            return tuple(event for event in run.events if event.sequence > after_sequence)

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """Subscribe to events and return an idempotent unsubscribe callback."""
        with self._lock:
            listener_id = self._next_listener_id
            self._next_listener_id += 1
            self._listeners[listener_id] = listener

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.pop(listener_id, None)

        return unsubscribe

    def _emit(
        self,
        run_id: str,
        *,
        event_type: TrainingEventType,
        expected_statuses: frozenset[str],
        status: TrainingStatus | None = None,
        stage: str | None = None,
        message: str | None = None,
        progress_percent: float | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        metrics: Mapping[str, float] | None = None,
        attributes: Mapping[str, TrainingScalar] | None = None,
        error_code: str | None = None,
    ) -> TrainingEvent:
        with self._lock:
            run = self._get_run(run_id)
            if run.snapshot.status not in expected_statuses:
                raise TrainingStateError(
                    f"cannot emit {event_type} while run {run_id} is {run.snapshot.status}"
                )
            event = TrainingEvent(
                run_id=run_id,
                sequence=len(run.events) + 1,
                event_type=event_type,
                status=status,
                stage=stage,
                message=message,
                progress_percent=progress_percent,
                step=step,
                total_steps=total_steps,
                metrics=dict(metrics or {}),
                attributes=dict(attributes or {}),
                error_code=error_code,
            )
            self._persist(event)
            run.events.append(event)
            run.snapshot = self._apply_event(run.snapshot, event)
            listeners = tuple(self._listeners.values())
        self._notify(event, listeners)
        return event

    def _persist(self, event: TrainingEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    @staticmethod
    def _apply_event(
        snapshot: TrainingRunSnapshot,
        event: TrainingEvent,
    ) -> TrainingRunSnapshot:
        """Derive the next snapshot from one already validated event."""
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
        return snapshot.model_copy(update=updates)

    def _get_run(self, run_id: str) -> _RunState:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise TrainingRunNotFoundError(run_id) from exc

    @staticmethod
    def _notify(event: TrainingEvent, listeners: tuple[EventListener, ...]) -> None:
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Observers are best-effort consumers and must not interrupt training.
                continue


__all__ = [
    "TrainingEventRecorder",
    "TrainingRunNotFoundError",
    "TrainingStateError",
]
