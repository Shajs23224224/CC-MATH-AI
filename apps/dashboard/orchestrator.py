"""Bounded, registry-driven training orchestration for the Training Dashboard (TD05)."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import BoundedSemaphore, Event, RLock
from typing import Protocol

from core.contracts.training import (
    TrainingJobSnapshot,
    TrainingProgress,
    TrainingProgressCallback,
    TrainingQueueSnapshot,
    TrainingScalar,
    TrainingTaskInfo,
)
from core.observability.training import (
    TrainingEventRecorder,
)

logger = logging.getLogger(__name__)

_TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")
_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
_ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING"})


class TrainingTaskHandler(Protocol):
    """Callable adapter for one registered model-training operation."""

    def __call__(
        self,
        parameters: Mapping[str, object],
        progress_callback: TrainingProgressCallback,
        cancellation: CancellationToken,
    ) -> object: ...


class TrainingTaskNotFoundError(KeyError):
    """Raised when a request names a task that is not allow-listed."""


class TrainingJobNotFoundError(KeyError):
    """Raised when a scheduler operation references an unknown job."""


class TrainingQueueFullError(RuntimeError):
    """Raised when all running and pending scheduler slots are occupied."""


class TrainingOrchestratorClosedError(RuntimeError):
    """Raised when new work is submitted after scheduler shutdown."""


class TrainingCancelledError(Exception):
    """Cooperative cancellation signal raised inside a training worker."""


class CancellationToken:
    """Thread-safe cancellation flag passed to registered task handlers.

    Training code should call raise_if_cancelled() at safe checkpoints in long
    loops. The progress callback supplied by the orchestrator also checks this
    token before recording progress, so instrumented engines stop at their next
    progress report. Python worker threads are never forcefully terminated.
    """

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise the scheduler's cancellation signal at a safe checkpoint."""
        if self.cancelled:
            raise TrainingCancelledError("training cancellation requested")


@dataclass(frozen=True, slots=True)
class _RegisteredTask:
    task_name: str
    description: str
    handler: TrainingTaskHandler


@dataclass(slots=True)
class _JobState:
    run_id: str
    task_name: str
    submitted_at: datetime
    token: CancellationToken
    parameters: dict[str, object]
    future: Future[object] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancellation_requested: bool = False


class TrainingTaskRegistry:
    """Explicit registry of task handlers available to the orchestration API."""

    def __init__(self) -> None:
        self._tasks: dict[str, _RegisteredTask] = {}
        self._lock = RLock()

    def register(
        self,
        task_name: str,
        handler: TrainingTaskHandler,
        *,
        description: str = "",
    ) -> None:
        """Register a task under a stable allow-listed identifier."""
        if not _TASK_NAME_PATTERN.fullmatch(task_name):
            raise ValueError("task_name must match [a-z][a-z0-9_.-]{0,79}")
        if len(description) > 500:
            raise ValueError("task description cannot exceed 500 characters")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            if task_name in self._tasks:
                raise ValueError(f"training task is already registered: {task_name}")
            self._tasks[task_name] = _RegisteredTask(
                task_name=task_name,
                description=description,
                handler=handler,
            )

    def unregister(self, task_name: str) -> bool:
        """Remove a task definition; already submitted jobs retain their handler."""
        with self._lock:
            return self._tasks.pop(task_name, None) is not None

    def resolve(self, task_name: str) -> TrainingTaskHandler:
        """Return a registered handler or raise a typed lookup error."""
        with self._lock:
            task = self._tasks.get(task_name)
        if task is None:
            raise TrainingTaskNotFoundError(task_name)
        return task.handler

    def list_tasks(self) -> tuple[TrainingTaskInfo, ...]:
        """Return public task names and descriptions without executable objects."""
        with self._lock:
            tasks = tuple(self._tasks.values())
        return tuple(
            TrainingTaskInfo(task_name=task.task_name, description=task.description)
            for task in sorted(tasks, key=lambda item: item.task_name)
        )


class TrainingOrchestrator:
    """Run registered training tasks in a bounded thread pool.

    The worker-pool capacity is max_workers + max_pending. Submissions beyond that
    capacity fail immediately rather than creating an unbounded in-memory queue.
    Run lifecycle events are written to the injected recorder, preserving the
    existing TD01/TD02/TD03/TD04 observability and dashboard flow.
    """

    def __init__(
        self,
        recorder: TrainingEventRecorder,
        *,
        registry: TrainingTaskRegistry | None = None,
        max_workers: int = 2,
        max_pending: int = 8,
        max_tracked_jobs: int = 10_000,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_pending < 0:
            raise ValueError("max_pending cannot be negative")
        if max_tracked_jobs < max_workers + max_pending:
            raise ValueError("max_tracked_jobs must cover the full scheduler capacity")

        self.recorder = recorder
        self.registry = registry or TrainingTaskRegistry()
        self.max_workers = max_workers
        self.max_pending = max_pending
        self.max_tracked_jobs = max_tracked_jobs
        self._capacity = max_workers + max_pending
        self._slots = BoundedSemaphore(self._capacity)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="training-worker",
        )
        self._jobs: dict[str, _JobState] = {}
        self._in_use = 0
        self._lock = RLock()
        self._closed = False

    def register_task(
        self,
        task_name: str,
        handler: TrainingTaskHandler,
        *,
        description: str = "",
    ) -> None:
        """Register a task handler with the orchestrator's allow-list."""
        self.registry.register(task_name, handler, description=description)

    def list_tasks(self) -> tuple[TrainingTaskInfo, ...]:
        """Expose the registered task catalogue."""
        return self.registry.list_tasks()

    def submit(
        self,
        task_name: str,
        *,
        name: str,
        parameters: Mapping[str, object] | None = None,
        metadata: Mapping[str, TrainingScalar] | None = None,
        run_id: str | None = None,
    ) -> TrainingJobSnapshot:
        """Create a QUEUED run and schedule its registered handler."""
        handler = self.registry.resolve(task_name)
        with self._lock:
            if self._closed:
                raise TrainingOrchestratorClosedError("training orchestrator is shut down")
            if not self._slots.acquire(blocking=False):
                raise TrainingQueueFullError(
                    f"training capacity is full ({self._capacity} running and pending slots)"
                )
            self._in_use += 1
            slot_transferred = False
            try:
                self._prune_terminal_jobs()
                if len(self._jobs) >= self.max_tracked_jobs:
                    raise TrainingQueueFullError(
                        "scheduler job history is at its configured limit"
                    )

                created = self.recorder.create_run(
                    name=name,
                    algorithm=task_name,
                    run_id=run_id,
                    metadata=dict(metadata or {}),
                )
                state = _JobState(
                    run_id=created.run_id,
                    task_name=task_name,
                    submitted_at=datetime.now(UTC),
                    token=CancellationToken(),
                    parameters=dict(parameters or {}),
                )
                self._jobs[created.run_id] = state
                try:
                    future = self._executor.submit(
                        self._execute,
                        created.run_id,
                        handler,
                        state.parameters,
                        state.token,
                    )
                except RuntimeError as exc:
                    self.recorder.fail(
                        created.run_id,
                        error_code="SCHEDULER_UNAVAILABLE",
                        message="Training task could not be scheduled",
                    )
                    state.finished_at = datetime.now(UTC)
                    raise TrainingOrchestratorClosedError(
                        "training executor is not accepting work"
                    ) from exc
                state.future = future
                # From this point the Future callback owns capacity release.
                slot_transferred = True
                future.add_done_callback(
                    lambda completed, job_id=created.run_id: self._on_future_done(
                        job_id, completed
                    )
                )
                return self._job_snapshot(state)
            except Exception:
                if not slot_transferred:
                    self._in_use -= 1
                    self._slots.release()
                raise

    def get_job(self, run_id: str) -> TrainingJobSnapshot:
        """Return scheduler details for one submitted job."""
        with self._lock:
            state = self._jobs.get(run_id)
            if state is None:
                raise TrainingJobNotFoundError(run_id)
            return self._job_snapshot(state)

    def list_jobs(self) -> tuple[TrainingJobSnapshot, ...]:
        """List tracked jobs newest first."""
        with self._lock:
            states = sorted(
                self._jobs.values(),
                key=lambda item: item.submitted_at,
                reverse=True,
            )
            return tuple(self._job_snapshot(state) for state in states)

    def queue_snapshot(self) -> TrainingQueueSnapshot:
        """Return current bounded queue and worker-pool counts."""
        with self._lock:
            statuses = [
                self.recorder.snapshot(run_id).status
                for run_id in self._jobs
            ]
            queued = statuses.count("QUEUED")
            running = statuses.count("RUNNING")
            return TrainingQueueSnapshot(
                max_workers=self.max_workers,
                max_pending=self.max_pending,
                capacity=self._capacity,
                queued=queued,
                running=running,
                in_use=self._in_use,
                available_slots=self._capacity - self._in_use,
                closed=self._closed,
            )

    def cancel(self, run_id: str) -> TrainingJobSnapshot:
        """Request cancellation; queued jobs stop immediately, running jobs cooperatively."""
        with self._lock:
            state = self._jobs.get(run_id)
            if state is None:
                raise TrainingJobNotFoundError(run_id)
            run = self.recorder.snapshot(run_id)
            if run.status in _TERMINAL_STATUSES:
                return self._job_snapshot(state)

            state.cancellation_requested = True
            state.token.cancel()
            future = state.future
            if run.status == "QUEUED" and future is not None:
                # Future.cancel() succeeds only when the worker has not started.
                # Its done callback records CANCELLED and releases the capacity slot.
                future.cancel()
            return self._job_snapshot(state)

    def wait(self, run_id: str, timeout: float | None = None) -> TrainingJobSnapshot:
        """Wait for a job's worker to finish; useful for tests and local orchestration."""
        with self._lock:
            state = self._jobs.get(run_id)
            if state is None:
                raise TrainingJobNotFoundError(run_id)
            future = state.future
        if future is not None:
            try:
                future.result(timeout=timeout)
            except CancelledError:
                pass
        return self.get_job(run_id)

    def shutdown(self, *, wait: bool = True, cancel_active: bool = True) -> None:
        """Stop accepting work and optionally request cancellation of active jobs."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active_ids = [
                run_id
                for run_id in self._jobs
                if self.recorder.snapshot(run_id).status in _ACTIVE_STATUSES
            ]
        if cancel_active:
            for run_id in active_ids:
                try:
                    self.cancel(run_id)
                except TrainingJobNotFoundError:
                    continue
        self._executor.shutdown(wait=wait, cancel_futures=cancel_active)

    def _execute(
        self,
        run_id: str,
        handler: TrainingTaskHandler,
        parameters: Mapping[str, object],
        token: CancellationToken,
    ) -> None:
        """Run one task and translate its outcome into recorder lifecycle events."""
        try:
            with self._lock:
                state = self._jobs[run_id]
                if token.cancelled:
                    self._cancel_run_locked(state)
                    return
                state.started_at = datetime.now(UTC)
                self.recorder.start_run(run_id, stage="initialization")

            recorder_callback = self.recorder.progress_callback(run_id)

            def report_progress(progress: TrainingProgress) -> None:
                token.raise_if_cancelled()
                recorder_callback(progress)

            token.raise_if_cancelled()
            handler(parameters, report_progress, token)
            with self._lock:
                state = self._jobs[run_id]
                if token.cancelled:
                    self._cancel_run_locked(state)
                else:
                    self.recorder.complete(run_id)
                    state.finished_at = datetime.now(UTC)
        except TrainingCancelledError:
            with self._lock:
                state = self._jobs[run_id]
                self._cancel_run_locked(state)
        except Exception:
            logger.exception(
                "training task failed (run_id=%s, task_name=%s)",
                run_id,
                self._jobs[run_id].task_name,
            )
            with self._lock:
                state = self._jobs[run_id]
                current = self.recorder.snapshot(run_id)
                if current.status in _ACTIVE_STATUSES:
                    try:
                        self.recorder.fail(
                            run_id,
                            error_code="TASK_EXECUTION_FAILED",
                            message="Training task failed; inspect server logs",
                            stage=current.current_stage,
                        )
                    except Exception:
                        logger.exception("could not record task failure (run_id=%s)", run_id)
                state.finished_at = datetime.now(UTC)

    def _cancel_run_locked(self, state: _JobState) -> None:
        """Record terminal cancellation once while scheduler state is locked."""
        current = self.recorder.snapshot(state.run_id)
        if current.status in _ACTIVE_STATUSES:
            self.recorder.cancel(state.run_id)
        state.finished_at = datetime.now(UTC)

    def _on_future_done(self, run_id: str, future: Future[object]) -> None:
        """Release bounded capacity and terminalize unexpected worker failures."""
        with self._lock:
            try:
                try:
                    error = future.exception()
                except CancelledError:
                    error = None
                state = self._jobs.get(run_id)
                if state is not None:
                    if future.cancelled():
                        self._cancel_run_locked(state)
                    elif error is not None:
                        current = self.recorder.snapshot(run_id)
                        if current.status in _ACTIVE_STATUSES:
                            try:
                                self.recorder.fail(
                                    run_id,
                                    error_code="WORKER_ABORTED",
                                    message="Training worker stopped unexpectedly",
                                    stage=current.current_stage,
                                )
                            except Exception:
                                logger.exception(
                                    "could not record aborted worker (run_id=%s)", run_id
                                )
                        state.finished_at = state.finished_at or datetime.now(UTC)
                    state.future = None
            except Exception:
                # A persistence failure must not leak a bounded scheduler slot.
                logger.exception("worker finalization failed (run_id=%s)", run_id)
                state = self._jobs.get(run_id)
                if state is not None:
                    state.future = None
                    state.finished_at = state.finished_at or datetime.now(UTC)
            finally:
                self._in_use -= 1
                self._slots.release()

    def _job_snapshot(self, state: _JobState) -> TrainingJobSnapshot:
        """Project current recorder state and scheduler timestamps into one response."""
        run = self.recorder.snapshot(state.run_id)
        queue_position: int | None = None
        if run.status == "QUEUED":
            queued = sorted(
                (
                    item
                    for item in self._jobs.values()
                    if self.recorder.snapshot(item.run_id).status == "QUEUED"
                ),
                key=lambda item: item.submitted_at,
            )
            queue_position = next(
                (index for index, item in enumerate(queued, start=1) if item.run_id == state.run_id),
                None,
            )
        finished_at = state.finished_at
        if run.status in _TERMINAL_STATUSES and finished_at is None:
            finished_at = run.ended_at or datetime.now(UTC)
            state.finished_at = finished_at
        return TrainingJobSnapshot(
            run_id=state.run_id,
            task_name=state.task_name,
            status=run.status,
            submitted_at=state.submitted_at,
            started_at=state.started_at or run.started_at,
            finished_at=finished_at,
            cancellation_requested=state.cancellation_requested,
            queue_position=queue_position,
        )

    def _prune_terminal_jobs(self) -> None:
        """Drop oldest scheduler metadata when the configured tracking limit is reached."""
        while len(self._jobs) >= self.max_tracked_jobs:
            removable = next(
                (
                    run_id
                    for run_id in self._jobs
                    if self.recorder.snapshot(run_id).status in _TERMINAL_STATUSES
                ),
                None,
            )
            if removable is None:
                return
            self._jobs.pop(removable, None)


__all__ = [
    "CancellationToken",
    "TrainingCancelledError",
    "TrainingJobNotFoundError",
    "TrainingOrchestrator",
    "TrainingOrchestratorClosedError",
    "TrainingQueueFullError",
    "TrainingTaskHandler",
    "TrainingTaskNotFoundError",
    "TrainingTaskRegistry",
]
