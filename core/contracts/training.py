"""Typed, immutable contracts for Training Dashboard observability (TD01)."""

from __future__ import annotations

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


class TrainingEvent(BaseModel):
    """One immutable, ordered event emitted by a training run."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1, max_length=100)
    sequence: int = Field(ge=1)
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
        if self.event_type == "RUN_CREATED" and self.status != "QUEUED":
            raise ValueError("RUN_CREATED events must have QUEUED status")
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
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("run timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.started_at is not None and self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.ended_at is not None and self.ended_at.tzinfo is None:
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
