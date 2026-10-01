from datetime import UTC, datetime

import pandas as pd

from core.contracts import DataQualityStatus
from data.quality import DataQualityEngine


def test_quality_detects_duplicates_and_invalid_values() -> None:
    frame = pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        "close": [100.0, None],
    })
    report = DataQualityEngine().check_frame(
        frame,
        dataset_id="prices",
        required_columns=("timestamp", "close"),
        key_columns=("timestamp",),
        numeric_columns=("close",),
        timestamp_column="timestamp",
    )
    codes = {issue.code for issue in report.issues}
    assert "DUPLICATES" in codes
    assert "INVALID_NUMERIC" in codes
    assert report.status == DataQualityStatus.INVALID


def test_quality_detects_future_data() -> None:
    frame = pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
        "close": [100.0, 101.0],
    })
    report = DataQualityEngine().check_frame(
        frame,
        dataset_id="prices",
        required_columns=("timestamp", "close"),
        numeric_columns=("close",),
        timestamp_column="timestamp",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert report.leakage_detected is True
    assert report.status == DataQualityStatus.INVALID
    assert report.is_usable is False


def test_quality_detects_stale_values() -> None:
    frame = pd.DataFrame({"value": [1.0, 1.0, 1.0, 1.0, 2.0]})
    report = DataQualityEngine().check_frame(
        frame,
        dataset_id="macro",
        required_columns=("value",),
        numeric_columns=("value",),
        stale_run_length=4,
    )
    assert any(issue.code == "STALE_VALUES" for issue in report.issues)
    assert report.status == DataQualityStatus.WARNING


def test_quality_detects_large_jumps() -> None:
    frame = pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        "close": [100.0, 250.0],
    })
    report = DataQualityEngine().check_frame(
        frame,
        dataset_id="prices",
        required_columns=("timestamp", "close"),
        numeric_columns=("close",),
        timestamp_column="timestamp",
        max_abs_return=1.0,
    )
    assert any(issue.code == "IMPOSSIBLE_JUMPS" for issue in report.issues)
    assert report.status == DataQualityStatus.INVALID


def test_quality_detects_robust_outlier() -> None:
    frame = pd.DataFrame({"value": [10, 11, 12, 13, 14, 100]})
    report = DataQualityEngine().check_frame(
        frame,
        dataset_id="macro",
        required_columns=("value",),
        numeric_columns=("value",),
        outlier_z_threshold=3.0,
        stale_run_length=None,
    )
    assert any(issue.code == "OUTLIERS" for issue in report.issues)


def test_quality_flags_survivorship_risk_without_universe_metadata() -> None:
    frame = pd.DataFrame({"symbol": ["AAPL"]})
    report = DataQualityEngine().check_frame(
        frame,
        dataset_id="universe",
        expected_symbols=("AAPL", "XYZ"),
        symbol_column="symbol",
        survivorship_metadata_present=False,
    )
    assert report.survivorship_bias_risk is True
    assert any(issue.code == "MISSING_UNIVERSE_MEMBERS" for issue in report.issues)
