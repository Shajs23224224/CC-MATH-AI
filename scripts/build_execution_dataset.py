"""D04: build a verified next-session execution dataset.

The label is a direction band around the next observed session's open-to-close
return. It is not a transaction-cost estimate and never becomes a model feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

EXPECTED_CLASSES = ("DOWN", "NEUTRAL", "UP")
REQUIRED_RAW_COLUMNS = ("open", "close")
REQUIRED_SNAPSHOT_TARGET = "target_next_return"
OUTPUT_COLUMNS = {
    "entry_date",
    "entry_open",
    "exit_close",
    "target_execution_return",
    "target_execution_direction",
}
FORBIDDEN_FEATURES = OUTPUT_COLUMNS | {
    "target_next_return",
    "target_next_direction",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _validate_threshold(threshold_bps: Real) -> float:
    if (
        isinstance(threshold_bps, bool)
        or not isinstance(threshold_bps, Real)
        or not math.isfinite(float(threshold_bps))
        or not 0.0 <= float(threshold_bps) <= 10_000.0
    ):
        raise ValueError("threshold_bps must be finite and between 0 and 10000")
    return float(threshold_bps)


def _slug(value: object) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()
    return result[:32] or "unknown"


def _verify_source_artifact(
    manifest: dict[str, Any],
    artifact_key: str,
    path: Path,
) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source manifest does not define an artifacts object")

    artifact = artifacts.get(artifact_key)
    if not isinstance(artifact, dict):
        raise ValueError(f"source manifest is missing {artifact_key}")

    expected_name = artifact.get("file")
    expected_hash = artifact.get("sha256")
    if expected_name != path.name:
        raise ValueError(f"{artifact_key} filename does not match source manifest")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"{artifact_key} SHA-256 is missing or malformed")

    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{artifact_key} SHA-256 verification failed")

    expected_bytes = artifact.get("bytes")
    if expected_bytes is not None:
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes != path.stat().st_size
        ):
            raise ValueError(f"{artifact_key} byte count does not match source manifest")
    return actual_hash


def _validate_datetime_index(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")
    if frame.index.isna().any():
        raise ValueError(f"{name} contains invalid dates")
    if frame.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} is not chronologically ordered")


def _numeric_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    description: str,
) -> np.ndarray:
    values = (
        frame.loc[:, columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=np.float64)
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{description} contains non-finite or non-numeric values")
    return values



def _class_coverage(labels: np.ndarray) -> dict[str, Any]:
    counts = {label: int(np.count_nonzero(labels == label)) for label in EXPECTED_CLASSES}
    missing = [label for label, count in counts.items() if count == 0]
    return {
        "rows": int(len(labels)),
        "class_counts": counts,
        "missing_classes": missing,
        "ready": not missing,
    }


def _assess_downstream_readiness(
    labels: np.ndarray,
    *,
    n_folds: int = 4,
    test_fraction: float = 0.10,
    calibration_fraction: float = 0.20,
) -> dict[str, Any]:
    """Check class coverage at the default D05/D06 expanding-window boundaries."""
    values = np.asarray(labels, dtype=str)
    global_coverage = _class_coverage(values)
    total_samples = len(values)
    test_size = math.floor(total_samples * test_fraction)
    initial_training_size = total_samples - n_folds * test_size
    setup_error = None
    if test_size < 2:
        setup_error = "Each outer test window must contain at least two rows."
    elif initial_training_size < 12:
        setup_error = "The initial expanding training prefix must contain at least 12 rows."

    d05_folds: list[dict[str, Any]] = []
    d06_folds: list[dict[str, Any]] = []
    if setup_error is None:
        for index in range(n_folds):
            test_start = initial_training_size + index * test_size
            test_end = test_start + test_size
            train_end = test_start

            training = _class_coverage(values[:train_end])
            test = _class_coverage(values[test_start:test_end])
            d05_folds.append(
                {
                    "fold": index + 1,
                    "train_start_index": 0,
                    "train_end_index": train_end,
                    "test_start_index": test_start,
                    "test_end_index": test_end,
                    "training": training,
                    "test": test,
                    "ready": training["ready"] and test["ready"],
                }
            )

            calibration_size = max(
                2,
                int(math.ceil(train_end * calibration_fraction)),
            )
            calibration_start = train_end - calibration_size
            calibration_fraction_for_estimator = (
                (calibration_size - 0.25) / train_end if train_end > 0 else float("nan")
            )
            calibration_split_representable = (
                calibration_start >= 12
                and 0.05 < calibration_fraction_for_estimator < 0.5
                and math.ceil(train_end * calibration_fraction_for_estimator) == calibration_size
            )

            outer_training = _class_coverage(values[:train_end])
            pre_calibration_training = _class_coverage(values[: max(0, calibration_start)])
            calibration = _class_coverage(values[max(0, calibration_start) : train_end])
            outer_test = _class_coverage(values[test_start:test_end])
            d06_folds.append(
                {
                    "fold": index + 1,
                    "train_start_index": 0,
                    "train_end_index": train_end,
                    "calibration_start_index": calibration_start,
                    "test_start_index": test_start,
                    "test_end_index": test_end,
                    "calibration_samples": calibration_size,
                    "calibration_holdout_fraction_representable": (calibration_split_representable),
                    "outer_training": outer_training,
                    "pre_calibration_training": pre_calibration_training,
                    "calibration": calibration,
                    "outer_test": outer_test,
                    "ready": (
                        outer_training["ready"]
                        and pre_calibration_training["ready"]
                        and calibration["ready"]
                        and outer_test["ready"]
                        and calibration_split_representable
                    ),
                }
            )

    global_missing = global_coverage["missing_classes"]
    d05_status = (
        "BLOCKED"
        if global_missing or (setup_error is None and not all(f["ready"] for f in d05_folds))
        else "INSUFFICIENT_DATA"
        if setup_error
        else "READY"
    )
    d06_status = (
        "BLOCKED"
        if global_missing or (setup_error is None and not all(f["ready"] for f in d06_folds))
        else "INSUFFICIENT_DATA"
        if setup_error
        else "READY"
    )
    profiles = {
        "D05": {
            "status": d05_status,
            "ready": d05_status == "READY",
            "configuration": {"n_folds": n_folds, "test_fraction": test_fraction},
            "reason": setup_error,
            "folds": d05_folds,
        },
        "D06": {
            "status": d06_status,
            "ready": d06_status == "READY",
            "configuration": {
                "n_folds": n_folds,
                "test_fraction": test_fraction,
                "calibration_fraction": calibration_fraction,
            },
            "reason": setup_error,
            "folds": d06_folds,
        },
    }
    status = (
        "READY"
        if d05_status == "READY" and d06_status == "READY"
        else "BLOCKED"
        if global_missing or d05_status == "BLOCKED" or d06_status == "BLOCKED"
        else "INSUFFICIENT_DATA"
    )
    return {
        "schema_version": "1.0",
        "status": status,
        "expected_classes": list(EXPECTED_CLASSES),
        "global_class_counts": global_coverage["class_counts"],
        "global_missing_classes": global_missing,
        "profiles": profiles,
        "interpretation": (
            "This preflight checks class coverage and temporal boundaries only. "
            "It does not guarantee estimator convergence, valid probabilities, "
            "predictive quality, or profitability."
        ),
    }


def build_execution_dataset(
    raw_path: Path,
    snapshot_path: Path,
    *,
    output_root: Path = Path("artifacts/execution_datasets"),
    threshold_bps: float = 10.0,
) -> dict[str, Any]:
    """Build one immutable, hash-manifested execution dataset.

    Feature row t is available after the close of session t. The theoretical
    execution is at the next observed OHLCV session's open and close.
    """
    threshold_bps = _validate_threshold(threshold_bps)
    raw_path = Path(raw_path).expanduser().resolve()
    snapshot_path = Path(snapshot_path).expanduser().resolve()

    if not raw_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("raw OHLCV or source snapshot does not exist")

    source_manifest_path = snapshot_path.parent / "dataset_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("source dataset_manifest.json does not exist")

    source_manifest = _read_json_object(
        source_manifest_path,
        "source dataset manifest",
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

    try:
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
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("raw OHLCV or source snapshot CSV is invalid") from exc

    if raw.empty or snapshot.empty:
        raise ValueError("raw OHLCV and source snapshot must not be empty")
    if not raw.columns.is_unique or not snapshot.columns.is_unique:
        raise ValueError("input CSV files must not contain duplicate column names")

    _validate_datetime_index(raw, "raw OHLCV")
    _validate_datetime_index(snapshot, "source snapshot")

    missing_prices = set(REQUIRED_RAW_COLUMNS) - set(raw.columns)
    if missing_prices:
        raise ValueError(f"raw OHLCV is missing columns: {sorted(missing_prices)}")

    dataset_info = source_manifest.get("dataset")
    if not isinstance(dataset_info, dict):
        raise ValueError("source manifest does not define dataset metadata")

    feature_names = dataset_info.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(name, str) or not name.strip() for name in feature_names)
    ):
        raise ValueError("source manifest must define a non-empty feature_names list")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("source manifest feature_names contains duplicates")

    forbidden = set(feature_names) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError(f"source features contain reserved/leaked columns: {sorted(forbidden)}")

    required_snapshot = [*feature_names, REQUIRED_SNAPSHOT_TARGET]
    missing_features = [name for name in required_snapshot if name not in snapshot.columns]
    if missing_features:
        raise ValueError(f"source snapshot is missing columns: {missing_features}")

    declared_rows = dataset_info.get("usable_feature_rows")
    if declared_rows is not None and (
        isinstance(declared_rows, bool)
        or not isinstance(declared_rows, int)
        or declared_rows != len(snapshot)
    ):
        raise ValueError("source snapshot row count does not match its manifest")

    try:
        positions = raw.index.get_indexer(snapshot.index)
    except (TypeError, ValueError) as exc:
        raise ValueError("source snapshot dates are incompatible with raw OHLCV dates") from exc

    if np.any(positions < 0):
        raise ValueError("some feature dates do not exist in raw OHLCV")
    if len(positions) == 0:
        raise ValueError("source snapshot is empty")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("feature dates must map to strictly increasing raw sessions")

    next_positions = positions + 1
    if np.any(next_positions >= len(raw)):
        raise ValueError("a feature date has no following observed OHLCV session")

    prices = raw.loc[:, list(REQUIRED_RAW_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    price_matrix = prices.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(price_matrix)):
        raise ValueError("raw OHLCV contains non-finite or non-numeric open/close prices")
    if np.any(price_matrix <= 0.0):
        raise ValueError("raw OHLCV contains non-positive open/close prices")

    feature_matrix = _numeric_matrix(
        snapshot,
        feature_names,
        "source features",
    )
    saved_target = pd.to_numeric(
        snapshot[REQUIRED_SNAPSHOT_TARGET],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(saved_target)):
        raise ValueError("source regression target contains non-finite values")

    close_values = prices["close"].to_numpy(dtype=np.float64)
    open_values = prices["open"].to_numpy(dtype=np.float64)
    next_close = close_values[next_positions]
    current_close = close_values[positions]
    next_open = open_values[next_positions]

    source_returns = next_close / current_close - 1.0
    if not np.allclose(
        saved_target,
        source_returns,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "source target_next_return does not match the next-session OHLCV chronology"
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
        ["UP", "DOWN"],
        default="NEUTRAL",
    )

    readiness = _assess_downstream_readiness(labels)

    entry_dates = raw.index.take(next_positions)
    if entry_dates.has_duplicates or not entry_dates.is_monotonic_increasing:
        raise ValueError("next-session entry dates must be unique and chronological")

    output = pd.DataFrame(
        feature_matrix,
        index=snapshot.index.copy(),
        columns=feature_names,
    )
    output.index.name = "feature_date"
    output["entry_date"] = entry_dates.strftime("%Y-%m-%d")
    output["entry_open"] = next_open
    output["exit_close"] = next_close
    output["target_execution_return"] = execution_returns
    output["target_execution_direction"] = labels

    ticker = str(source_manifest.get("experiment", {}).get("ticker", "UNKNOWN"))
    safe_ticker = _slug(ticker)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(UTC)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"{safe_ticker}_execution_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    dataset_path = run_dir / "execution_dataset.csv"
    manifest_path = run_dir / "execution_dataset_manifest.json"
    _write_csv_atomic(output, dataset_path)

    class_counts = {label: int(np.sum(labels == label)) for label in EXPECTED_CLASSES}
    class_shares = {label: count / len(labels) for label, count in class_counts.items()}

    manifest = {
        "schema_version": "1.0",
        "contract_version": "D04.3",
        "created_at": created_at.isoformat(),
        "experiment": {
            "ticker": ticker,
            "threshold_bps": threshold_bps,
            "threshold_fraction": threshold,
            "threshold_is_transaction_cost": False,
        },
        "source": {
            "raw_ohlcv_file": raw_path.name,
            "raw_ohlcv_sha256": raw_hash,
            "raw_ohlcv_bytes": raw_path.stat().st_size,
            "source_snapshot_file": snapshot_path.name,
            "source_snapshot_sha256": snapshot_hash,
            "source_snapshot_bytes": snapshot_path.stat().st_size,
            "source_manifest_file": source_manifest_path.name,
            "source_manifest_sha256": _sha256_file(source_manifest_path),
            "source_manifest_bytes": source_manifest_path.stat().st_size,
        },
        "execution_assumption": {
            "signal_available_after": "feature session close",
            "entry": "next observed session open",
            "exit": "same next session close",
            "return_formula": "next_session_close / next_session_open - 1",
            "return_type": "gross, before commissions, spread and slippage",
            "entry_price_assumption": "theoretical fill at recorded open",
            "feature_rule": "only source manifest feature_names are model inputs",
        },
        "target": {
            "name": "target_execution_return",
            "classification_name": "target_execution_direction",
            "label_rule": (
                "UP if return > threshold; DOWN if return < -threshold; NEUTRAL otherwise"
            ),
            "classes": list(EXPECTED_CLASSES),
            "threshold_units": "basis_points",
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
            "direction_share": class_shares,
            "execution_return_mean": float(np.mean(execution_returns)),
            "execution_return_std": float(np.std(execution_returns, ddof=1))
            if len(execution_returns) > 1
            else 0.0,
            "execution_return_min": float(np.min(execution_returns)),
            "execution_return_max": float(np.max(execution_returns)),
        },
        "data_readiness": readiness,
        "serialization": {
            "csv_float_format": "%.17g",
            "csv_date_format": "%Y-%m-%dT%H:%M:%S.%f%z",
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
            "The threshold is a classification label band, not a transaction-cost estimate.",
            "The recorded opening and closing prices are theoretical execution assumptions.",
            "Commissions, spread, slippage, market impact, funding and taxes are not modeled.",
            "The source manifest is unsigned; SHA-256 detects changes "
            "but does not authenticate origin.",
            "Historical predictive metrics do not establish future profitability.",
        ],
    }
    _write_json_atomic(manifest_path, manifest)

    return {
        "status": "COMPLETED",
        "run_id": run_dir.name,
        "output_directory": str(run_dir),
        "rows": len(output),
        "features": len(feature_names),
        "threshold_bps": threshold_bps,
        "direction_distribution": class_counts,
        "readiness_status": readiness["status"],
        "d05_readiness_status": readiness["profiles"]["D05"]["status"],
        "d06_readiness_status": readiness["profiles"]["D06"]["status"],
        "execution_dataset_sha256": manifest["artifact"]["sha256"],
        "source_manifest_sha256": manifest["source"]["source_manifest_sha256"],
        "manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a verified next-session open-to-close execution dataset."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_datasets"),
    )
    parser.add_argument("--threshold-bps", type=float, default=10.0)
    parser.add_argument(
        "--require-d06-ready",
        action="store_true",
        help=(
            "Return exit code 2 unless the default D06 temporal/class-coverage profile is ready."
        ),
    )
    args = parser.parse_args()

    result = build_execution_dataset(
        args.raw,
        args.snapshot,
        output_root=args.output_root,
        threshold_bps=args.threshold_bps,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.require_d06_ready and result["d06_readiness_status"] != "READY":
        print(
            "D06 readiness requirement failed; inspect data_readiness in the manifest.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
