"""D04 tests: causal execution labels, strict source identity and immutable outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_execution_dataset import build_execution_dataset


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_source(tmp_path, closes=None):
    if closes is None:
        closes = [100.0, 100.3, 99.7, 100.0, 100.05, 100.0]

    dates = pd.bdate_range("2025-01-02", periods=len(closes))
    raw = pd.DataFrame(
        {
            "open": np.full(len(closes), 100.0),
            "high": np.asarray(closes, dtype=float) + 0.2,
            "low": np.asarray(closes, dtype=float) - 0.2,
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

    raw_path = tmp_path / "raw_ohlcv.csv"
    snapshot_path = tmp_path / "dataset_snapshot.csv"
    raw.to_csv(raw_path, index_label="date", float_format="%.17g")
    snapshot.to_csv(snapshot_path, index_label="date", float_format="%.17g")

    manifest = {
        "schema_version": "1.0",
        "experiment": {"ticker": "TEST", "random_seed": 28},
        "artifacts": {
            "raw_ohlcv": {
                "file": raw_path.name,
                "sha256": digest(raw_path),
                "bytes": raw_path.stat().st_size,
            },
            "dataset_snapshot": {
                "file": snapshot_path.name,
                "sha256": digest(snapshot_path),
                "bytes": snapshot_path.stat().st_size,
            },
        },
        "dataset": {
            "feature_names": ["feature_a", "feature_b"],
            "usable_feature_rows": len(snapshot),
        },
    }
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return raw_path, snapshot_path


def output_from(result):
    return Path(result["output_directory"]) / "execution_dataset.csv"


def test_execution_target_uses_next_observed_session_open_and_close(tmp_path):
    raw_path, snapshot_path = create_source(tmp_path)
    result = build_execution_dataset(
        raw_path,
        snapshot_path,
        output_root=tmp_path / "outputs",
        threshold_bps=10.0,
    )
    output_path = output_from(result)
    output = pd.read_csv(output_path, index_col="feature_date")

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
        [100.3, 99.7, 100.0, 100.05, 100.0]
    )
    assert output["target_execution_return"].iloc[0] == pytest.approx(0.003)
    assert output["target_execution_return"].iloc[1] == pytest.approx(-0.003)
    assert result["direction_distribution"] == {
        "DOWN": 1,
        "NEUTRAL": 3,
        "UP": 1,
    }


def test_manifest_records_integrity_and_execution_assumptions(tmp_path):
    raw_path, snapshot_path = create_source(tmp_path)
    result = build_execution_dataset(raw_path, snapshot_path, output_root=tmp_path / "out")
    run_dir = Path(result["output_directory"])
    dataset_path = run_dir / "execution_dataset.csv"
    manifest_path = run_dir / "execution_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["contract_version"] == "D04.2"
    assert manifest["experiment"]["threshold_bps"] == 10.0
    assert manifest["experiment"]["threshold_is_transaction_cost"] is False
    assert manifest["target"]["classes"] == ["DOWN", "NEUTRAL", "UP"]
    assert manifest["execution_assumption"]["entry"] == "next observed session open"
    assert manifest["dataset"]["rows"] == 5
    assert manifest["artifact"]["sha256"] == digest(dataset_path)
    assert manifest["artifact"]["bytes"] == dataset_path.stat().st_size
    assert manifest["source"]["raw_ohlcv_sha256"] == digest(raw_path)
    assert manifest["source"]["source_snapshot_sha256"] == digest(snapshot_path)
    assert manifest["source"]["source_manifest_sha256"] == digest(
        tmp_path / "dataset_manifest.json"
    )


def test_source_hash_and_byte_count_mismatches_fail_closed(tmp_path):
    raw_path, snapshot_path = create_source(tmp_path)
    with raw_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="SHA-256"):
        build_execution_dataset(raw_path, snapshot_path, output_root=tmp_path / "out")


def test_source_manifest_rejects_incorrect_row_count(tmp_path):
    raw_path, snapshot_path = create_source(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["usable_feature_rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="row count"):
        build_execution_dataset(raw_path, snapshot_path, output_root=tmp_path / "out")


def test_reserved_future_or_target_columns_cannot_be_features(tmp_path):
    raw_path, snapshot_path = create_source(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["feature_names"].append("entry_open")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="reserved"):
        build_execution_dataset(raw_path, snapshot_path, output_root=tmp_path / "out")


def test_source_target_must_match_adjacent_ohlcv_sessions(tmp_path):
    raw_path, snapshot_path = create_source(tmp_path)
    frame = pd.read_csv(snapshot_path, index_col="date", parse_dates=True)
    frame.iloc[0, frame.columns.get_loc("target_next_return")] += 0.01
    frame.to_csv(snapshot_path, index_label="date", float_format="%.17g")

    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["dataset_snapshot"]["sha256"] = digest(snapshot_path)
    manifest["artifacts"]["dataset_snapshot"]["bytes"] = snapshot_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="target_next_return"):
        build_execution_dataset(raw_path, snapshot_path, output_root=tmp_path / "out")


@pytest.mark.parametrize("threshold", [-0.1, 10_001.0, float("nan"), float("inf"), True])
def test_invalid_threshold_is_rejected(tmp_path, threshold):
    raw_path, snapshot_path = create_source(tmp_path)
    with pytest.raises(ValueError, match="threshold_bps"):
        build_execution_dataset(
            raw_path,
            snapshot_path,
            output_root=tmp_path / "out",
            threshold_bps=threshold,
        )


def test_feature_date_without_following_session_is_rejected(tmp_path):
    raw_path, snapshot_path = create_source(tmp_path)
    raw = pd.read_csv(raw_path, index_col="date", parse_dates=True)
    snapshot = pd.read_csv(snapshot_path, index_col="date", parse_dates=True)
    last_date = raw.index[-1]
    snapshot.loc[last_date] = [1.0, 0.1, 0.0, "NEUTRAL"]
    snapshot.to_csv(snapshot_path, index_label="date", float_format="%.17g")

    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["usable_feature_rows"] = len(snapshot)
    manifest["artifacts"]["dataset_snapshot"]["sha256"] = digest(snapshot_path)
    manifest["artifacts"]["dataset_snapshot"]["bytes"] = snapshot_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="following observed"):
        build_execution_dataset(raw_path, snapshot_path, output_root=tmp_path / "out")
