from __future__ import annotations

import pandas as pd

from core.contracts import MacroObservation
from core.errors import DataQualityError
from .common import (
    normalize_dataframe_columns,
    normalize_date,
    normalize_identifier,
    normalize_macro_frequency,
    normalize_unit,
)


_REQUIRED_COLUMNS = (
    "series_id",
    "name",
    "geography",
    "period_end",
    "released_at",
    "value",
    "unit",
    "frequency",
    "source",
)


def normalize_macro_frame(frame: pd.DataFrame) -> tuple[MacroObservation, ...]:
    work = normalize_dataframe_columns(frame)
    missing = [column for column in _REQUIRED_COLUMNS if column not in work.columns]
    if missing:
        raise DataQualityError(f"missing macro columns: {missing}")

    work["period_end"] = pd.to_datetime(work["period_end"], errors="coerce", utc=True).dt.date
    work["released_at"] = pd.to_datetime(work["released_at"], errors="coerce", utc=True).dt.date
    work["value"] = pd.to_numeric(work["value"], errors="coerce")

    if work["period_end"].isna().any() or work["released_at"].isna().any():
        raise DataQualityError("macro data contains invalid dates")
    if work["value"].isna().any():
        raise DataQualityError("macro data contains invalid numeric values")
    for column in _REQUIRED_COLUMNS:
        if work[column].isna().any():
            raise DataQualityError(f"macro data contains missing values in {column!r}")

    duplicate_columns = ["series_id", "geography", "period_end", "released_at"]
    if "tenor" in work.columns:
        duplicate_columns.append("tenor")
    if work.duplicated(subset=duplicate_columns).any():
        raise DataQualityError("duplicate macro observations detected")

    work = work.sort_values(["released_at", "period_end", "series_id"]).reset_index(drop=True)

    observations: list[MacroObservation] = []
    for row in work.itertuples(index=False):
        tenor = getattr(row, "tenor", None) if "tenor" in work.columns else None
        observations.append(
            MacroObservation(
                series_id=normalize_identifier(str(row.series_id)),
                name=str(row.name).strip(),
                geography=normalize_identifier(str(row.geography)),
                period_end=normalize_date(row.period_end),
                released_at=normalize_date(row.released_at),
                value=float(row.value),
                unit=normalize_unit(str(row.unit)),
                frequency=normalize_macro_frequency(str(row.frequency)),
                source=str(row.source).strip(),
                tenor=None if pd.isna(tenor) else str(tenor).strip(),
            )
        )

    return tuple(observations)
