"""Local dashboard application components."""

from apps.dashboard.event_store import (
    CorruptTrainingEventStoreError,
    TrainingEventStore,
    TrainingEventStoreError,
)

__all__ = [
    "CorruptTrainingEventStoreError",
    "TrainingEventStore",
    "TrainingEventStoreError",
]
