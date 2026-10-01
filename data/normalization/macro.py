from __future__ import annotations

from datetime import date

import pandas as pd

from core.contracts import MacroObservation
from core.errors import DataQualityError


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
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataQualityError(f"missing macro columns: {missing}")

    work = frame.copy()
    for column in ("period_end", "released_at"):
        work[column] = pd.to_datetime(work[column], errors="coerce", utc=True).dt.date
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
        observation = MacroObservation(
            series_id=str(row.series_id),
            name=str(row.name),
            geography=str(row.geography),
            period_end=_as_date(row.period_end),
            released_at=_as_date(row.released_at),
            value=float(row.value),
            unit=str(row.unit),
            frequency=str(row.frequency).lower(),
            source=str(row.source),
            tenor=None if pd.isna(tenor) else str(tenor),
        )
        observations.append(observation)

    return tuple(observations)


def _as_date(value: object) -> date:
    if not isinstance(value, date):
        raise DataQualityError("macro date has invalid type")
    return value
