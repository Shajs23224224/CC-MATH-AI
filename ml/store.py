"""Append-only JSONL persistence for experiment metadata (no executable model artifacts)."""

from __future__ import annotations

from pathlib import Path

from core.contracts.ml import ExperimentRecord
from .errors import ExperimentPersistenceError


class JSONLExperimentStore:
    """Small append-only record store suitable for CPU-first local experiments.

    Store configuration and metrics only. Trained estimators are deliberately not
    pickled or loaded as executable artifacts.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> None:
        """Append a validated JSON record, creating its parent directory if needed."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
        except (OSError, ValueError) as exc:
            raise ExperimentPersistenceError("could not append experiment record") from exc

    def read_all(self) -> tuple[ExperimentRecord, ...]:
        """Recover and validate every record; corrupted rows fail explicitly."""
        if not self.path.exists():
            return ()
        records: list[ExperimentRecord] = []
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(ExperimentRecord.model_validate_json(line))
                    except ValueError as exc:
                        raise ExperimentPersistenceError(
                            f"invalid experiment record at line {line_number}"
                        ) from exc
        except OSError as exc:
            raise ExperimentPersistenceError("could not read experiment store") from exc
        return tuple(records)


__all__ = ["JSONLExperimentStore"]
