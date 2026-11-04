"""Shared, hash-manifested persistence for real-market training datasets."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_snapshot(frame: pd.DataFrame, path: Path) -> str:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=True,
        index_label="date",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
        lineterminator="\n",
    )
    temporary.replace(path)
    return _sha256_file(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def persist_market_dataset_artifacts(
    run_dir: Path,
    history: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    ticker: str,
    source: str,
    period_requested: str | None,
    seed: int,
    feature_names: Sequence[str],
    provider_version: str | None = None,
) -> dict[str, Any]:
    """Persist OHLCV, causal features/targets, and the common D04 source manifest.

    The Dashboard and standalone training CLI share this exact persistence
    contract, so either run can be consumed by scripts/build_execution_dataset.py.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if history.empty or rows.empty:
        raise ValueError("Cannot persist an empty historical dataset")
    if not history.columns.is_unique or not rows.columns.is_unique:
        raise ValueError("Historical and feature frames must have unique columns")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Historical OHLCV must have a DatetimeIndex")
    if not isinstance(rows.index, pd.DatetimeIndex):
        raise ValueError("Dataset snapshot must have a DatetimeIndex")
    for frame, description in (
        (history, "Historical OHLCV"),
        (rows, "Dataset snapshot"),
    ):
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{description} dates must be unique and chronological")
    if not rows.index.isin(history.index).all():
        raise ValueError("Every feature date must exist in the historical OHLCV index")

    missing_raw = {"open", "close"} - set(history.columns)
    if missing_raw:
        raise ValueError(f"Historical OHLCV is missing columns: {sorted(missing_raw)}")

    names = tuple(feature_names)
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("feature_names must be a non-empty unique sequence of names")
    missing_features = set(names) - set(rows.columns)
    if missing_features:
        raise ValueError(f"Dataset snapshot is missing features: {sorted(missing_features)}")
    if "target_next_return" not in rows.columns:
        raise ValueError("Dataset snapshot is missing target_next_return")
    if "target_next_direction" in rows.columns:
        raise ValueError("Dataset snapshot must not predefine target_next_direction")

    feature_values = (
        rows.loc[:, list(names)]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=np.float64)
    )
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("Dataset features contain non-finite or non-numeric values")

    target = pd.to_numeric(
        rows["target_next_return"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise ValueError("Dataset targets contain non-finite or non-numeric values")

    raw_prices = (
        history.loc[:, ["open", "close"]]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=np.float64)
    )
    if not np.all(np.isfinite(raw_prices)) or np.any(raw_prices <= 0.0):
        raise ValueError("Historical open/close prices must be finite and positive")

    raw_path = run_dir / "raw_ohlcv.csv"
    snapshot_path = run_dir / "dataset_snapshot.csv"
    manifest_path = run_dir / "dataset_manifest.json"
    for path in (raw_path, snapshot_path, manifest_path):
        if path.exists() or path.with_name(path.name + ".tmp").exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing reproducibility artifact: {path}"
            )

    snapshot = rows.copy(deep=True)
    snapshot["target_next_direction"] = np.where(target > 0.0, "UP", "DOWN")

    raw_hash = _write_csv_snapshot(history, raw_path)
    snapshot_hash = _write_csv_snapshot(snapshot, snapshot_path)

    total_samples = len(snapshot)
    test_samples = max(2, int(np.ceil(total_samples * 0.20)))
    training_samples = total_samples - test_samples
    inner_validation_samples = max(
        2,
        int(np.ceil(training_samples * 0.15)),
    )
    inner_training_samples = training_samples - inner_validation_samples

    direction_counts = {
        str(label): int(count)
        for label, count in (snapshot["target_next_direction"].value_counts().sort_index().items())
    }
    target_series = pd.Series(target, dtype=np.float64)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": {
            "ticker": ticker,
            "period_requested": period_requested,
            "random_seed": seed,
        },
        "artifacts": {
            "raw_ohlcv": {
                "file": raw_path.name,
                "sha256": raw_hash,
                "bytes": raw_path.stat().st_size,
            },
            "dataset_snapshot": {
                "file": snapshot_path.name,
                "sha256": snapshot_hash,
                "bytes": snapshot_path.stat().st_size,
            },
        },
        "dataset": {
            "source": source,
            "downloaded_rows": len(history),
            "usable_feature_rows": total_samples,
            "first_ohlcv_date": str(history.index[0]),
            "last_ohlcv_date": str(history.index[-1]),
            "first_feature_date": str(snapshot.index[0]),
            "last_feature_date": str(snapshot.index[-1]),
            "feature_names": list(names),
            "regression_target": "next_session_return = close[t+1]/close[t] - 1",
            "classification_target": "UP if next_session_return > 0 else DOWN",
            "direction_distribution": direction_counts,
            "regression_target_statistics": {
                "mean": float(target_series.mean()),
                "std": float(target_series.std(ddof=0)),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
            },
        },
        "temporal_split": {
            "method": "chronological",
            "test_fraction": 0.20,
            "training_samples": training_samples,
            "test_samples": test_samples,
            "train_start_index": 0,
            "train_end_index": training_samples,
            "test_start_index": training_samples,
            "test_end_index": total_samples,
            "test_used_for_selection": False,
            "boosting_inner_validation_fraction": 0.15,
            "boosting_inner_validation_samples": inner_validation_samples,
            "boosting_inner_training_samples": inner_training_samples,
        },
        "serialization": {
            "csv_float_format": "%.17g",
            "csv_date_format": "%Y-%m-%dT%H:%M:%S.%f%z",
            "index_column": "date",
            "read_csv_float_precision": "round_trip",
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_full_version": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "yfinance": provider_version or "not_used",
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest
