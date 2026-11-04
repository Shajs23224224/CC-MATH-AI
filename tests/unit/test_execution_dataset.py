"""Tests for verified next-session execution dataset construction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_execution_dataset import build_execution_dataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_source(tmp_path, closes):
    dates = pd.bdate_range("2025-01-02", periods=len(closes))

    raw = pd.DataFrame(
        {
            "open": np.full(len(closes), 100.0),
            "high": np.maximum(np.asarray(closes), 100.0) + 0.1,
            "low": np.minimum(np.asarray(closes), 100.0) - 0.1,
            "close": np.asarray(closes, dtype=float),
            "volume": np.full(len(closes), 100_000.0),
        },
        index=dates,
    )
    raw.index.name = "date"

    snapshot = pd.DataFrame(
        {
            "feature_a": np.arange(len(closes) - 1, dtype=float),
            "feature_b": np.arange(len(closes) - 1, dtype=float) / 10.0,
        },
        index=dates[:-1],
    )
    snapshot.index.name = "date"

    snapshot["target_next_return"] = (
        raw["close"].shift(-1).iloc[:-1].to_numpy()
        / raw["close"].iloc[:-1].to_numpy()
        - 1.0
    )
    snapshot["target_next_direction"] = np.where(
        snapshot["target_next_return"] > 0.0,
        "UP",
        "DOWN",
    )

    raw_path = tmp_path / "raw.csv"
    snapshot_path = tmp_path / "snapshot.csv"

    raw.to_csv(
        raw_path,
        index=True,
        index_label="date",
        float_format="%.17g",
    )
    snapshot.to_csv(
        snapshot_path,
        index=True,
        index_label="date",
        float_format="%.17g",
    )

    source_manifest = {
        "schema_version": "1.0",
        "experiment": {
            "ticker": "TEST",
            "random_seed": 28,
        },
        "artifacts": {
            "raw_ohlcv": {
                "file": raw_path.name,
                "sha256": digest(raw_path),
            },
            "dataset_snapshot": {
                "file": snapshot_path.name,
                "sha256": digest(snapshot_path),
            },
        },
        "dataset": {
            "feature_names": ["feature_a", "feature_b"],
            "usable_feature_rows": len(snapshot),
        },
    }

    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )

    return raw_path, snapshot_path


def test_execution_target_uses_next_session_open_and_close(tmp_path):
    raw_path, snapshot_path = create_source(
        tmp_path,
        [100.0, 100.3, 99.7, 100.0, 100.1, 100.0],
    )

    result = build_execution_dataset(
        raw_path,
        snapshot_path,
        output_root=tmp_path / "outputs",
        threshold_bps=10.0,
    )

    output_dir = tmp_path / "outputs" / Path(result["output_directory"]).name
    output = pd.read_csv(
        output_dir / "execution_dataset.csv",
        index_col="feature_date",
    )

    assert len(output) == 5
    assert output["target_execution_direction"].tolist() == [
        "UP",
        "DOWN",
        "NEUTRAL",
        "NEUTRAL",
        "NEUTRAL",
    ]

    assert output["entry_open"].tolist() == pytest.approx([100.0] * 5)
    assert output["exit_close"].tolist() == pytest.approx(
        [100.3, 99.7, 100.0, 100.1, 100.0]
    )

    assert output["target_execution_return"].iloc[0] == pytest.approx(0.003)
    assert output["target_execution_return"].iloc[1] == pytest.approx(-0.003)

    assert result["direction_distribution"] == {
        "UP": 1,
        "DOWN": 1,
        "NEUTRAL": 3,
    }


def test_source_hash_mismatch_fails_closed(tmp_path):
    raw_path, snapshot_path = create_source(
        tmp_path,
        [100.0, 100.2, 99.8, 100.1, 100.3],
    )

    with raw_path.open("a", encoding="utf-8") as file:
        file.write("\n")

    with pytest.raises(ValueError, match="SHA-256"):
        build_execution_dataset(
            raw_path,
            snapshot_path,
            output_root=tmp_path / "outputs",
        )


def test_negative_threshold_is_rejected(tmp_path):
    raw_path, snapshot_path = create_source(
        tmp_path,
        [100.0, 100.2, 99.8, 100.1, 100.3],
    )

    with pytest.raises(ValueError, match="threshold_bps"):
        build_execution_dataset(
            raw_path,
            snapshot_path,
            output_root=tmp_path / "outputs",
            threshold_bps=-1.0,
        )


def test_manifest_records_gross_execution_assumption(tmp_path):
    raw_path, snapshot_path = create_source(
        tmp_path,
        [100.0, 100.2, 99.8, 100.1, 100.3],
    )

    result = build_execution_dataset(
        raw_path,
        snapshot_path,
        output_root=tmp_path / "outputs",
    )

    manifest_path = tmp_path / "outputs" / Path(
        result["output_directory"]
    ).name / "execution_dataset_manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["experiment"]["threshold_bps"] == 10.0
    assert manifest["experiment"]["threshold_is_transaction_cost"] is False
    assert manifest["execution_assumption"]["return_type"].startswith("gross")
    assert manifest["dataset"]["rows"] == 4
    assert manifest["artifact"]["sha256"] == digest(
        manifest_path.parent / "execution_dataset.csv"
    )
