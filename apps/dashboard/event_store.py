"""Append-only JSONL persistence for Training Dashboard events (TD01)."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from pydantic import ValidationError

from core.contracts.training import TrainingEvent


class TrainingEventStoreError(RuntimeError):
    """Raised when training events cannot be persisted or recovered."""


class CorruptTrainingEventStoreError(TrainingEventStoreError):
    """Raised when an existing JSONL row cannot be validated as a training event."""


class TrainingEventStore:
    """Persist immutable events as one validated JSON object per line.

    The store is designed for the dashboard's single local application process.
    A per-instance lock protects concurrent threads; multi-process writers are not
    supported by this first version.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def append(self, event: TrainingEvent) -> None:
        """Append one event and flush it before returning."""
        if not isinstance(event, TrainingEvent):
            raise TypeError("event must be a TrainingEvent instance")
        try:
            serialized = event.model_dump_json() + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(serialized)
                    stream.flush()
        except (OSError, ValueError) as exc:
            raise TrainingEventStoreError("could not append training event") from exc

    def read_events(
        self,
        *,
        run_id: str | None = None,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> tuple[TrainingEvent, ...]:
        """Read validated events, optionally filtering by run and sequence cursor."""
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        if not self.path.exists():
            return ()

        recovered: list[TrainingEvent] = []
        try:
            with self._lock, self.path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = TrainingEvent.model_validate_json(line)
                    except (ValidationError, ValueError) as exc:
                        raise CorruptTrainingEventStoreError(
                            f"invalid training event at line {line_number}"
                        ) from exc
                    if run_id is not None and event.run_id != run_id:
                        continue
                    if event.sequence <= after_sequence:
                        continue
                    recovered.append(event)
                    if limit is not None and len(recovered) >= limit:
                        break
        except OSError as exc:
            raise TrainingEventStoreError("could not read training event store") from exc
        return tuple(recovered)

    def read_all(self) -> tuple[TrainingEvent, ...]:
        """Recover every event in append order."""
        return self.read_events()


__all__ = [
    "CorruptTrainingEventStoreError",
    "TrainingEventStore",
    "TrainingEventStoreError",
]
