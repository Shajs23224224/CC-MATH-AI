"""Tests for versioned, non-executable experiment persistence."""

from pathlib import Path

import pytest

from core.contracts.ml import ExperimentRecord
from ml import JSONLExperimentStore
from ml.errors import ExperimentPersistenceError


def test_experiment_store_round_trip(tmp_path: Path) -> None:
    record = ExperimentRecord(
        kind="classical_ml_evaluation",
        subject_id="ridge:test",
        status="EXPERIMENTAL",
        configuration={"seed": 28, "split": "chronological"},
        results={"rmse": 0.25, "test_samples": 12},
        limitations=("synthetic fixture",),
    )
    store = JSONLExperimentStore(tmp_path / "nested" / "experiments.jsonl")
    store.append(record)
    recovered = store.read_all()
    assert recovered == (record,)


def test_corrupted_experiment_line_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    path.write_text('{"not":"an experiment"}\n', encoding="utf-8")
    with pytest.raises(ExperimentPersistenceError, match="line 1"):
        JSONLExperimentStore(path).read_all()
