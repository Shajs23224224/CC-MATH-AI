"""Tests for exact historical and derived dataset persistence."""

import hashlib
import json

import numpy as np
import pandas as pd

from apps.dashboard.training_tasks import (
    _persist_reproducibility_artifacts,
)
from scripts.train_real_models import FEATURE_COLUMNS


def test_reproducibility_snapshots_round_trip_exactly(tmp_path):
    dates = pd.date_range("2024-01-02", periods=32, freq="B")

    history = pd.DataFrame(
        {
            "open": np.linspace(100.1234567890123, 105.9876543210987, 32),
            "high": np.linspace(101.1234567890123, 106.9876543210987, 32),
            "low": np.linspace(99.1234567890123, 104.9876543210987, 32),
            "close": np.linspace(100.5234567890123, 105.4876543210987, 32),
            "volume": np.arange(1_000_000, 1_000_032, dtype=np.int64),
        },
        index=dates,
    )

    rows = pd.DataFrame(
        {
            name: np.linspace(index + 0.123456789012345, index + 1.987654321098765, 32)
            for index, name in enumerate(FEATURE_COLUMNS)
        },
        index=dates,
    )

    rows["target_next_return"] = np.linspace(-0.02, 0.025, 32)

    manifest = _persist_reproducibility_artifacts(
        tmp_path,
        history,
        rows,
        ticker="SPY",
        period="10y",
        seed=28,
    )

    raw_path = tmp_path / "raw_ohlcv.csv"
    snapshot_path = tmp_path / "dataset_snapshot.csv"
    manifest_path = tmp_path / "dataset_manifest.json"

    raw_loaded = pd.read_csv(
        raw_path,
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")

    snapshot_loaded = pd.read_csv(
        snapshot_path,
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")

    np.testing.assert_array_equal(
        raw_loaded[history.columns].to_numpy(),
        history.to_numpy(),
    )

    expected_snapshot = rows.copy()
    expected_snapshot["target_next_direction"] = np.where(
        expected_snapshot["target_next_return"].to_numpy() > 0,
        "UP",
        "DOWN",
    )

    np.testing.assert_array_equal(
        snapshot_loaded[list(FEATURE_COLUMNS) + ["target_next_return"]].to_numpy(
            dtype=np.float64
        ),
        expected_snapshot[list(FEATURE_COLUMNS) + ["target_next_return"]].to_numpy(
            dtype=np.float64
        ),
    )

    assert snapshot_loaded["target_next_direction"].tolist() == (
        expected_snapshot["target_next_direction"].tolist()
    )

    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    assert manifest["artifacts"]["raw_ohlcv"]["sha256"] == raw_hash
    assert manifest["artifacts"]["dataset_snapshot"]["sha256"] == snapshot_hash

    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == manifest

    assert manifest["dataset"]["usable_feature_rows"] == 32
    assert manifest["temporal_split"]["test_samples"] == 7
    assert manifest["temporal_split"]["training_samples"] == 25
    assert manifest["dataset"]["direction_distribution"]["DOWN"] == 14
    assert manifest["dataset"]["direction_distribution"]["UP"] == 18


def test_reproducibility_snapshot_refuses_to_overwrite(tmp_path):
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    history = pd.DataFrame(
        {
            "open": np.arange(10.0) + 100,
            "high": np.arange(10.0) + 101,
            "low": np.arange(10.0) + 99,
            "close": np.arange(10.0) + 100.5,
            "volume": np.arange(10.0) + 1000,
        },
        index=dates,
    )
    rows = pd.DataFrame(
        {name: np.arange(10.0) for name in FEATURE_COLUMNS},
        index=dates,
    )
    rows["target_next_return"] = np.linspace(-0.01, 0.01, 10)

    _persist_reproducibility_artifacts(
        tmp_path,
        history,
        rows,
        ticker="SPY",
        period="2y",
        seed=28,
    )

    import pytest

    with pytest.raises(FileExistsError):
        _persist_reproducibility_artifacts(
            tmp_path,
            history,
            rows,
            ticker="SPY",
            period="2y",
            seed=28,
        )
