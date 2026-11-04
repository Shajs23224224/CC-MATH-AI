"""Tests for the shared Dashboard/CLI market-data artifact contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.utils.market_dataset_artifacts import persist_market_dataset_artifacts
from scripts.build_execution_dataset import build_execution_dataset
from scripts.run_execution_decision_policy import _load_validated_d04_input
from scripts.run_execution_ml_walk_forward import load_execution_dataset


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_source():
    row_count = 120
    dates = pd.bdate_range("2024-01-02", periods=row_count + 1)
    step = np.arange(row_count + 1, dtype=np.float64)
    close = np.full(row_count + 1, 100.0, dtype=np.float64)
    execution_returns = np.tile(
        np.asarray([-0.002, 0.0, 0.002], dtype=np.float64),
        row_count // 3,
    )
    open_prices = np.full(row_count + 1, 100.0, dtype=np.float64)
    open_prices[1:] = close[1:] / (1.0 + execution_returns)
    history = pd.DataFrame(
        {
            "open": open_prices,
            "high": np.maximum(open_prices, close) * 1.01,
            "low": np.minimum(open_prices, close) * 0.99,
            "close": close,
            "volume": 1_000_000.0 + step * 100.0,
        },
        index=dates,
    )
    rows = pd.DataFrame(
        {
            "feature_a": step[:-1],
            "feature_b": np.cos(step[:-1]),
        },
        index=dates[:-1],
    )
    rows["target_next_return"] = (
        history["close"].shift(-1).iloc[:-1].to_numpy()
        / history["close"].iloc[:-1].to_numpy()
        - 1.0
    )
    return history, rows


def test_shared_artifacts_are_round_trip_safe_and_consumable_by_d04(tmp_path):
    history, rows = _synthetic_source()
    manifest = persist_market_dataset_artifacts(
        tmp_path,
        history,
        rows,
        ticker="SYNTH",
        source="synthetic test OHLCV",
        period_requested=None,
        seed=28,
        feature_names=("feature_a", "feature_b"),
        provider_version="test",
    )

    raw_path = tmp_path / "raw_ohlcv.csv"
    snapshot_path = tmp_path / "dataset_snapshot.csv"
    manifest_path = tmp_path / "dataset_manifest.json"
    raw_loaded = pd.read_csv(
        raw_path,
        index_col="date",
        parse_dates=True,
        float_precision="round_trip",
    )
    snapshot_loaded = pd.read_csv(
        snapshot_path,
        index_col="date",
        parse_dates=True,
        float_precision="round_trip",
    )

    np.testing.assert_array_equal(
        raw_loaded.loc[:, history.columns].to_numpy(),
        history.to_numpy(),
    )
    np.testing.assert_array_equal(
        snapshot_loaded.loc[:, ["feature_a", "feature_b", "target_next_return"]].to_numpy(),
        rows.loc[:, ["feature_a", "feature_b", "target_next_return"]].to_numpy(),
    )
    assert manifest["artifacts"]["raw_ohlcv"]["sha256"] == _digest(raw_path)
    assert manifest["artifacts"]["dataset_snapshot"]["sha256"] == _digest(snapshot_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    result = build_execution_dataset(
        raw_path,
        snapshot_path,
        output_root=tmp_path / "execution",
    )
    assert result["status"] == "COMPLETED"
    assert result["rows"] == len(rows)
    assert result["readiness_status"] == "READY"
    assert result["d05_readiness_status"] == "READY"
    assert result["d06_readiness_status"] == "READY"

    d05_dataset, d05_dates, d05_manifest = load_execution_dataset(output_path)
    d06_dataset, d06_dates, d06_manifest = _load_validated_d04_input(output_path)

    assert len(d05_dataset.features) == len(rows)
    assert d05_dataset.target == d06_dataset.target
    assert d05_dates == d06_dates
    assert d05_manifest["contract_version"] == "D04.3"
    assert d06_manifest["target"]["classes"] == ["DOWN", "NEUTRAL", "UP"]


def test_shared_persistence_refuses_to_overwrite_artifacts(tmp_path):
    history, rows = _synthetic_source()
    kwargs = {
        "ticker": "SYNTH",
        "source": "synthetic test OHLCV",
        "period_requested": None,
        "seed": 28,
        "feature_names": ("feature_a", "feature_b"),
    }
    persist_market_dataset_artifacts(tmp_path, history, rows, **kwargs)

    with pytest.raises(FileExistsError):
        persist_market_dataset_artifacts(tmp_path, history, rows, **kwargs)


def test_shared_persistence_rejects_non_finite_features(tmp_path):
    history, rows = _synthetic_source()
    rows.loc[rows.index[0], "feature_a"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        persist_market_dataset_artifacts(
            tmp_path,
            history,
            rows,
            ticker="SYNTH",
            source="synthetic test OHLCV",
            period_requested=None,
            seed=28,
            feature_names=("feature_a", "feature_b"),
        )
