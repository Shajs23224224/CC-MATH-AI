"""Build an auditable next-session open-to-close execution dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_artifact(
    manifest: dict[str, Any],
    artifact_key: str,
    path: Path,
) -> str:
    artifact = manifest.get("artifacts", {}).get(artifact_key, {})
    expected_name = artifact.get("file")
    expected_hash = artifact.get("sha256")

    if expected_name != path.name:
        raise ValueError(
            f"{artifact_key} filename does not match source manifest"
        )

    if not expected_hash:
        raise ValueError(f"{artifact_key} SHA-256 is missing from source manifest")

    actual_hash = _sha256_file(path)

    if actual_hash != expected_hash:
        raise ValueError(f"{artifact_key} SHA-256 verification failed")

    return actual_hash


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=True,
        index_label="feature_date",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
        lineterminator="\n",
    )
    temporary.replace(path)


def build_execution_dataset(
    raw_path: Path,
    snapshot_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_datasets"),
    threshold_bps: float = 10.0,
) -> dict[str, Any]:
    """Build a three-state target using next-session open-to-close returns."""
    if (
        isinstance(threshold_bps, bool)
        or not math.isfinite(threshold_bps)
        or not 0.0 <= threshold_bps <= 10_000.0
    ):
        raise ValueError("threshold_bps must be finite and between 0 and 10000")

    raw_path = raw_path.expanduser().resolve()
    snapshot_path = snapshot_path.expanduser().resolve()

    if not raw_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("raw OHLCV or source snapshot does not exist")

    source_manifest_path = snapshot_path.parent / "dataset_manifest.json"

    if not source_manifest_path.is_file():
        raise FileNotFoundError("source dataset_manifest.json does not exist")

    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )

    raw_hash = _verify_source_artifact(
        source_manifest,
        "raw_ohlcv",
        raw_path,
    )
    snapshot_hash = _verify_source_artifact(
        source_manifest,
        "dataset_snapshot",
        snapshot_path,
    )

    raw = pd.read_csv(
        raw_path,
        index_col="date",
        parse_dates=True,
        float_precision="round_trip",
    )
    snapshot = pd.read_csv(
        snapshot_path,
        index_col="date",
        parse_dates=True,
        float_precision="round_trip",
    )

    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("raw OHLCV must have a DatetimeIndex")
    if not isinstance(snapshot.index, pd.DatetimeIndex):
        raise ValueError("source snapshot must have a DatetimeIndex")

    for name, frame in (("raw OHLCV", raw), ("snapshot", snapshot)):
        if frame.index.has_duplicates:
            raise ValueError(f"{name} contains duplicate dates")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} is not chronologically ordered")

    required_prices = {"open", "close"}
    missing_prices = required_prices - set(raw.columns)

    if missing_prices:
        raise ValueError(f"raw OHLCV is missing columns: {sorted(missing_prices)}")

    dataset_info = source_manifest.get("dataset", {})
    feature_names = dataset_info.get("feature_names", [])

    if not feature_names:
        raise ValueError("source manifest does not define feature_names")

    required_snapshot = feature_names + ["target_next_return"]
    missing_features = [
        name for name in required_snapshot if name not in snapshot.columns
    ]

    if missing_features:
        raise ValueError(
            f"source snapshot is missing columns: {missing_features}"
        )

    prices = raw[["open", "close"]].apply(pd.to_numeric, errors="coerce")
    features = snapshot[feature_names].apply(pd.to_numeric, errors="coerce")
    saved_target = pd.to_numeric(
        snapshot["target_next_return"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    price_matrix = prices.to_numpy(dtype=np.float64)
    feature_matrix = features.to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(price_matrix)):
        raise ValueError("raw OHLCV contains non-finite open or close prices")
    if np.any(price_matrix <= 0.0):
        raise ValueError("raw OHLCV contains non-positive prices")

    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("source features contain non-finite values")
    if not np.all(np.isfinite(saved_target)):
        raise ValueError("source regression target contains non-finite values")

    # Locate each feature date in the original OHLCV index.
    positions = raw.index.get_indexer(snapshot.index)

    if np.any(positions < 0):
        raise ValueError("some feature dates do not exist in raw OHLCV")

    if len(positions) == 0:
        raise ValueError("source snapshot is empty")

    next_positions = positions + 1

    if np.any(next_positions >= len(raw)):
        raise ValueError(
            "a snapshot feature date has no following OHLCV session"
        )

    # Verify the previous close-to-close objective before replacing it.
    current_close = prices["close"].to_numpy(dtype=np.float64)[positions]
    next_close = prices["close"].to_numpy(dtype=np.float64)[next_positions]
    next_open = prices["open"].to_numpy(dtype=np.float64)[next_positions]

    recalculated_old_target = next_close / current_close - 1.0

    if not np.allclose(
        saved_target,
        recalculated_old_target,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "source target_next_return does not match the OHLCV chronology"
        )

    execution_returns = next_close / next_open - 1.0

    if not np.all(np.isfinite(execution_returns)):
        raise ValueError("execution returns contain non-finite values")

    threshold = threshold_bps / 10_000.0

    labels = np.select(
        [
            execution_returns > threshold,
            execution_returns < -threshold,
        ],
        [
            "UP",
            "DOWN",
        ],
        default="NEUTRAL",
    )

    entry_dates = raw.index.take(next_positions)

    # Only the original causal features are exposed as model inputs.
    output = features.copy()
    output.index.name = "feature_date"
    output["entry_date"] = entry_dates.strftime("%Y-%m-%d")
    output["entry_open"] = next_open
    output["exit_close"] = next_close
    output["target_execution_return"] = execution_returns
    output["target_execution_direction"] = labels

    ticker = str(
        source_manifest.get("experiment", {}).get("ticker", "UNKNOWN")
    )
    safe_ticker = re.sub(r"[^a-zA-Z0-9_-]+", "_", ticker).lower()

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / (
        f"{safe_ticker}_execution_{timestamp}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)

    dataset_path = run_dir / "execution_dataset.csv"
    manifest_path = run_dir / "execution_dataset_manifest.json"

    _write_csv_atomic(output, dataset_path)

    class_counts = {
        label: int(np.sum(labels == label))
        for label in ("UP", "DOWN", "NEUTRAL")
    }

    distribution = {
        label: count / len(labels)
        for label, count in class_counts.items()
    }

    output_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": {
            "ticker": ticker,
            "threshold_bps": threshold_bps,
            "threshold_fraction": threshold,
            "threshold_is_transaction_cost": False,
        },
        "source": {
            "raw_ohlcv_file": raw_path.name,
            "raw_ohlcv_sha256": raw_hash,
            "source_snapshot_file": snapshot_path.name,
            "source_snapshot_sha256": snapshot_hash,
            "source_manifest_file": source_manifest_path.name,
            "source_manifest_sha256": _sha256_file(source_manifest_path),
        },
        "execution_assumption": {
            "signal_available_after": "feature session close",
            "entry": "next observed session open",
            "exit": "same next session close",
            "return_formula": "next_session_close / next_session_open - 1",
            "return_type": "gross, before commissions, spread and slippage",
            "entry_price_assumption": "theoretical fill at recorded open",
            "feature_rule": "only source snapshot feature_names are model inputs",
        },
        "target": {
            "name": "target_execution_return",
            "classification_name": "target_execution_direction",
            "label_rule": (
                "UP if return > threshold; DOWN if return < -threshold; "
                "NEUTRAL otherwise"
            ),
            "classes": ["DOWN", "NEUTRAL", "UP"],
        },
        "dataset": {
            "rows": len(output),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "first_feature_date": output.index[0].isoformat(),
            "last_feature_date": output.index[-1].isoformat(),
            "first_entry_date": str(output["entry_date"].iloc[0]),
            "last_entry_date": str(output["entry_date"].iloc[-1]),
            "direction_distribution": class_counts,
            "direction_share": distribution,
            "execution_return_mean": float(np.mean(execution_returns)),
            "execution_return_std": float(np.std(execution_returns, ddof=1)),
            "execution_return_min": float(np.min(execution_returns)),
            "execution_return_max": float(np.max(execution_returns)),
        },
        "serialization": {
            "csv_float_format": "%.17g",
            "index_column": "feature_date",
            "read_csv_float_precision": "round_trip",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifact": {
            "file": dataset_path.name,
            "sha256": _sha256_file(dataset_path),
            "bytes": dataset_path.stat().st_size,
        },
        "limitations": [
            "This dataset does not model commissions, spread or slippage.",
            "The recorded opening price is a theoretical execution assumption.",
            "The threshold is a target-label band, not a verified broker cost.",
            "Historical predictive metrics do not establish future profitability.",
        ],
    }

    _write_json_atomic(manifest_path, output_manifest)

    return {
        "status": "COMPLETED",
        "output_directory": str(run_dir),
        "rows": len(output),
        "features": len(feature_names),
        "threshold_bps": threshold_bps,
        "direction_distribution": class_counts,
        "execution_dataset_sha256": output_manifest["artifact"]["sha256"],
        "manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a verified next-session execution dataset."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_datasets"),
    )
    parser.add_argument("--threshold-bps", type=float, default=10.0)

    args = parser.parse_args()

    result = build_execution_dataset(
        args.raw,
        args.snapshot,
        output_root=args.output_root,
        threshold_bps=args.threshold_bps,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
