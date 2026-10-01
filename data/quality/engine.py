from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from core.contracts import (
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
    DataQualityStatus,
)


class DataQualityEngine:
    """Deterministic data-quality checks for tabular financial datasets."""

    def check_frame(
        self,
        frame: pd.DataFrame,
        *,
        dataset_id: str,
        required_columns: Sequence[str] = (),
        key_columns: Sequence[str] = (),
        numeric_columns: Sequence[str] = (),
        timestamp_column: str | None = None,
        as_of: datetime | None = None,
        max_abs_return: float | None = None,
        outlier_z_threshold: float | None = 8.0,
        stale_run_length: int | None = 5,
        expected_symbols: Sequence[str] | None = None,
        symbol_column: str = "symbol",
        survivorship_metadata_present: bool = True,
    ) -> DataQualityReport:
        work = frame.copy()
        issues: list[DataQualityIssue] = []
        leakage_detected = False
        survivorship_risk = False

        missing_columns = [c for c in required_columns if c not in work.columns]
        if missing_columns:
            issues.append(
                DataQualityIssue(
                    code="MISSING_COLUMNS",
                    severity=DataQualitySeverity.ERROR,
                    message=f"required columns are missing: {missing_columns}",
                    count=len(missing_columns),
                )
            )
            return self._report(
                dataset_id,
                len(work),
                0,
                len(work),
                issues,
                leakage_detected=True,
                survivorship_risk=False,
            )

        if key_columns:
            duplicate_count = int(work.duplicated(subset=list(key_columns), keep=False).sum())
            if duplicate_count:
                issues.append(
                    DataQualityIssue(
                        code="DUPLICATES",
                        severity=DataQualitySeverity.ERROR,
                        message="duplicate key records detected",
                        count=duplicate_count,
                    )
                )

        for column in numeric_columns:
            numeric = pd.to_numeric(work[column], errors="coerce")
            invalid = int(numeric.isna().sum())
            if invalid:
                issues.append(
                    DataQualityIssue(
                        code="INVALID_NUMERIC",
                        severity=DataQualitySeverity.ERROR,
                        message=f"non-numeric or missing values in {column}",
                        field=column,
                        count=invalid,
                    )
                )

            if outlier_z_threshold is not None:
                outlier_count = _robust_outlier_count(numeric, outlier_z_threshold)
                if outlier_count:
                    issues.append(
                        DataQualityIssue(
                            code="OUTLIERS",
                            severity=DataQualitySeverity.WARNING,
                            message=f"robust outliers detected in {column}",
                            field=column,
                            count=outlier_count,
                        )
                    )

            if stale_run_length is not None and stale_run_length > 1:
                stale_count = _stale_value_count(numeric, stale_run_length)
                if stale_count:
                    issues.append(
                        DataQualityIssue(
                            code="STALE_VALUES",
                            severity=DataQualitySeverity.WARNING,
                            message=f"repeated identical values detected in {column}",
                            field=column,
                            count=stale_count,
                        )
                    )

        if timestamp_column:
            timestamps = pd.to_datetime(work[timestamp_column], errors="coerce", utc=True)
            invalid_timestamps = int(timestamps.isna().sum())
            if invalid_timestamps:
                issues.append(
                    DataQualityIssue(
                        code="INVALID_TIMESTAMPS",
                        severity=DataQualitySeverity.ERROR,
                        message=f"invalid timestamps in {timestamp_column}",
                        field=timestamp_column,
                        count=invalid_timestamps,
                    )
                )
            elif as_of is not None:
                future_count = int((timestamps > _as_utc(as_of)).sum())
                if future_count:
                    leakage_detected = True
                    issues.append(
                        DataQualityIssue(
                            code="FUTURE_DATA",
                            severity=DataQualitySeverity.ERROR,
                            message="records occur after requested as_of",
                            field=timestamp_column,
                            count=future_count,
                        )
                    )

            if max_abs_return is not None and max_abs_return > 0:
                for column in numeric_columns:
                    series = pd.to_numeric(work[column], errors="coerce")
                    jumps = series.pct_change().abs()
                    jump_count = int((jumps > max_abs_return).sum())
                    if jump_count:
                        issues.append(
                            DataQualityIssue(
                                code="IMPOSSIBLE_JUMPS",
                                severity=DataQualitySeverity.ERROR,
                                message=f"absolute change exceeds threshold in {column}",
                                field=column,
                                count=jump_count,
                            )
                        )

        if expected_symbols is not None:
            if symbol_column not in work.columns:
                issues.append(
                    DataQualityIssue(
                        code="UNIVERSE_COLUMN_MISSING",
                        severity=DataQualitySeverity.WARNING,
                        message=f"cannot check expected universe because {symbol_column} is absent",
                        field=symbol_column,
                    )
                )
            else:
                observed = {str(v).strip().upper() for v in work[symbol_column].dropna()}
                expected = {str(v).strip().upper() for v in expected_symbols}
                missing_symbols = expected - observed
                if missing_symbols:
                    survivorship_risk = not survivorship_metadata_present
                    issues.append(
                        DataQualityIssue(
                            code="MISSING_UNIVERSE_MEMBERS",
                            severity=(
                                DataQualitySeverity.WARNING
                                if survivorship_metadata_present
                                else DataQualitySeverity.ERROR
                            ),
                            message="expected universe members are absent; review delistings/survivorship",
                            field=symbol_column,
                            count=len(missing_symbols),
                        )
                    )

        error_count = sum(
            issue.count for issue in issues if issue.severity == DataQualitySeverity.ERROR
        )
        warning_count = sum(
            issue.count for issue in issues if issue.severity == DataQualitySeverity.WARNING
        )

        status = (
            DataQualityStatus.INVALID
            if leakage_detected or error_count
            else DataQualityStatus.WARNING
            if warning_count
            else DataQualityStatus.VALID
        )
        rejected = min(len(work), error_count)
        accepted = max(0, len(work) - rejected)

        return self._report(
            dataset_id,
            len(work),
            accepted,
            rejected,
            issues,
            leakage_detected=leakage_detected,
            survivorship_risk=survivorship_risk,
            status=status,
        )

    @staticmethod
    def _report(
        dataset_id: str,
        rows: int,
        accepted: int,
        rejected: int,
        issues: list[DataQualityIssue],
        *,
        leakage_detected: bool,
        survivorship_risk: bool,
        status: DataQualityStatus,
    ) -> DataQualityReport:
        return DataQualityReport(
            dataset_id=dataset_id,
            checked_at=datetime.now(timezone.utc),
            rows=rows,
            accepted_rows=accepted,
            rejected_rows=rejected,
            status=status,
            issues=tuple(issues),
            leakage_detected=leakage_detected,
            survivorship_bias_risk=survivorship_risk,
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _robust_outlier_count(series: pd.Series, threshold: float) -> int:
    clean = series.dropna()
    if len(clean) < 5:
        return 0
    median = float(clean.median())
    mad = float(np.median(np.abs(clean.to_numpy() - median)))
    if mad == 0:
        return 0
    modified_z = 0.6745 * (clean - median).abs() / mad
    return int((modified_z > threshold).sum())


def _stale_value_count(series: pd.Series, run_length: int) -> int:
    equal_previous = series.eq(series.shift(1))
    groups = (~equal_previous).cumsum()
    run_sizes = series.groupby(groups, dropna=False).transform("size")
    return int((equal_previous & (run_sizes >= run_length)).sum())
