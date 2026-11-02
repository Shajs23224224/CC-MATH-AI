"""Typed, immutable contracts for Training Dashboard observability (TD01)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

TrainingStatus = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
TrainingEventType = Literal[
    "RUN_CREATED",
    "STATE_CHANGED",
    "PROGRESS",
    "METRICS",
    "WARNING",
    "ERROR",
]
TrainingScalar = str | int | float | bool | None


class TrainingProgress(BaseModel):
    """One algorithm-native progress update emitted by an instrumented training engine."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    stage: str = Field(min_length=1, max_length=120)
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    step: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    attributes: dict[str, TrainingScalar] = Field(default_factory=dict)
    message: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_progress_shape(self) -> TrainingProgress:
        """Require meaningful progress and consistent counters."""
        if not self.stage.strip():
            raise ValueError("stage must not be blank")
        if (self.step is None) != (self.total_steps is None):
            raise ValueError("step and total_steps must be supplied together")
        if self.step is not None and self.total_steps is not None and self.step > self.total_steps:
            raise ValueError("step cannot exceed total_steps")
        if self.progress_percent is None and self.step is None:
            raise ValueError("progress requires a percentage or step counters")
        if any(not key.strip() for key in self.metrics):
            raise ValueError("metric names must not be blank")
        if any(not key.strip() for key in self.attributes):
            raise ValueError("attribute names must not be blank")
        return self


TrainingProgressCallback = Callable[[TrainingProgress], None]


class TrainingEvent(BaseModel):
    """One immutable, ordered event emitted by a training run."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1, max_length=100)
    sequence: int = Field(ge=1)
    run_name: str | None = Field(default=None, max_length=160)
    algorithm: str | None = Field(default=None, max_length=120)
    run_metadata: dict[str, TrainingScalar] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: TrainingEventType
    status: TrainingStatus | None = None
    stage: str | None = Field(default=None, max_length=120)
    message: str | None = Field(default=None, max_length=2_000)
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    step: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    attributes: dict[str, TrainingScalar] = Field(default_factory=dict)
    error_code: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_event_shape(self) -> TrainingEvent:
        """Require enough information for each event kind and consistent progress fields."""
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        if (self.step is None) != (self.total_steps is None):
            raise ValueError("step and total_steps must be supplied together")
        if self.step is not None and self.total_steps is not None and self.step > self.total_steps:
            raise ValueError("step cannot exceed total_steps")
        if self.event_type == "RUN_CREATED":
            if (
                self.status != "QUEUED"
                or self.run_name is None
                or not self.run_name.strip()
                or self.algorithm is None
                or not self.algorithm.strip()
            ):
                raise ValueError("RUN_CREATED events require QUEUED status, run_name and algorithm")
        elif self.run_name is not None or self.algorithm is not None or self.run_metadata:
            raise ValueError("run metadata is only allowed on RUN_CREATED events")
        if self.event_type == "STATE_CHANGED" and self.status is None:
            raise ValueError("STATE_CHANGED events require a status")
        if self.event_type == "PROGRESS":
            if self.progress_percent is None and self.step is None:
                raise ValueError("PROGRESS events require a percentage or step counters")
        if self.event_type == "METRICS" and not self.metrics:
            raise ValueError("METRICS events require at least one metric")
        if self.event_type == "WARNING" and not self.message:
            raise ValueError("WARNING events require a message")
        if self.event_type == "ERROR":
            if self.status != "FAILED" or not self.error_code or not self.message:
                raise ValueError("ERROR events require FAILED status, error_code and message")
        if any(not key.strip() for key in self.metrics):
            raise ValueError("metric names must not be blank")
        if any(not key.strip() for key in self.attributes):
            raise ValueError("attribute names must not be blank")
        return self


class TrainingRunSnapshot(BaseModel):
    """Current immutable view of one run, derived from its ordered event stream."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    run_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=160)
    algorithm: str = Field(min_length=1, max_length=120)
    status: TrainingStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    event_count: int = Field(ge=1)
    current_stage: str | None = Field(default=None, max_length=120)
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    current_step: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=1)
    latest_metrics: dict[str, float] = Field(default_factory=dict)
    error_code: str | None = Field(default=None, max_length=120)
    message: str | None = Field(default=None, max_length=2_000)
    metadata: dict[str, TrainingScalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_snapshot_state(self) -> TrainingRunSnapshot:
        """Keep timestamps and counters coherent with the run lifecycle."""
        if self.created_at.utcoffset() is None or self.updated_at.utcoffset() is None:
            raise ValueError("run timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if not self.name.strip() or not self.algorithm.strip():
            raise ValueError("run name and algorithm must not be blank")
        if self.started_at is not None and self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")
        if self.ended_at is not None and self.ended_at.utcoffset() is None:
            raise ValueError("ended_at must be timezone-aware")
        if self.status == "QUEUED" and self.started_at is not None:
            raise ValueError("queued runs cannot have started_at")
        if self.status == "RUNNING" and self.started_at is None:
            raise ValueError("running runs require started_at")
        if self.status in {"COMPLETED", "FAILED", "CANCELLED"} and self.ended_at is None:
            raise ValueError("terminal runs require ended_at")
        if (self.current_step is None) != (self.total_steps is None):
            raise ValueError("current_step and total_steps must be supplied together")
        if (
            self.current_step is not None
            and self.total_steps is not None
            and self.current_step > self.total_steps
        ):
            raise ValueError("current_step cannot exceed total_steps")
        return self


class TrainingTaskSubmission(BaseModel):
    """Validated API payload for submitting one registered training task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    name: str = Field(min_length=1, max_length=160)
    parameters: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, TrainingScalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_submission(self) -> TrainingTaskSubmission:
        """Reject blank labels before a run enters the queue."""
        if not self.name.strip():
            raise ValueError("name must not be blank")
        if any(not key.strip() for key in self.parameters):
            raise ValueError("parameter names must not be blank")
        if any(not key.strip() for key in self.metadata):
            raise ValueError("metadata names must not be blank")
        return self


class TrainingTaskInfo(BaseModel):
    """Public description of one registered, allow-listed training task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class TrainingJobSnapshot(BaseModel):
    """Scheduler state associated with one recorder-backed training run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=100)
    task_name: str = Field(min_length=1, max_length=80)
    status: TrainingStatus
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancellation_requested: bool = False
    queue_position: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_job_timestamps(self) -> TrainingJobSnapshot:
        """Require timezone-aware, chronologically ordered scheduler timestamps."""
        timestamps = (self.submitted_at, self.started_at, self.finished_at)
        if any(value is not None and value.utcoffset() is None for value in timestamps):
            raise ValueError("job timestamps must be timezone-aware")
        if self.started_at is not None and self.started_at < self.submitted_at:
            raise ValueError("started_at cannot precede submitted_at")
        if self.finished_at is not None:
            reference = self.started_at or self.submitted_at
            if self.finished_at < reference:
                raise ValueError("finished_at cannot precede job start or submission")
        if self.status == "QUEUED" and self.started_at is not None:
            raise ValueError("queued jobs cannot have started_at")
        if self.status == "RUNNING" and self.started_at is None:
            raise ValueError("running jobs require started_at")
        if self.status in {"COMPLETED", "FAILED", "CANCELLED"} and self.finished_at is None:
            raise ValueError("terminal jobs require finished_at")
        if self.status != "QUEUED" and self.queue_position is not None:
            raise ValueError("only queued jobs may have a queue position")
        return self


class TrainingQueueSnapshot(BaseModel):
    """Bounded worker-pool capacity and current queue counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_workers: int = Field(ge=1)
    max_pending: int = Field(ge=0)
    capacity: int = Field(ge=1)
    queued: int = Field(ge=0)
    running: int = Field(ge=0)
    in_use: int = Field(ge=0)
    available_slots: int = Field(ge=0)
    closed: bool

    @model_validator(mode="after")
    def validate_capacity(self) -> TrainingQueueSnapshot:
        """Keep queue counts within the configured bounded worker-pool capacity."""
        if self.capacity != self.max_workers + self.max_pending:
            raise ValueError("capacity must equal max_workers plus max_pending")
        if self.in_use > self.capacity:
            raise ValueError("in_use cannot exceed scheduler capacity")
        if self.queued + self.running > self.in_use:
            raise ValueError("queued and running jobs cannot exceed reserved capacity")
        if self.available_slots != self.capacity - self.in_use:
            raise ValueError("available_slots must match unreserved scheduler capacity")
        return self
