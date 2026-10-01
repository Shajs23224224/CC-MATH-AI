from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from core.contracts import MacroDataRequest
from core.errors import DataProviderError

_REQUIRED_COLUMNS = frozenset(
    {"series_id", "name", "geography", "period_end", "released_at", "value", "unit", "frequency"}
)
_COLUMN_ALIASES = {
    "series": "series_id",
    "series_code": "series_id",
    "indicator": "name",
    "country": "geography",
    "region": "geography",
    "period": "period_end",
    "release_date": "released_at",
    "release": "released_at",
    "publication_date": "released_at",
    "tenor_name": "tenor",
}


class LocalCSVMacroProvider:
    """Local CSV macro provider for development and point-in-time testing."""

    name = "local_csv_macro"

    def __init__(self, root: str | Path = "data/local/macro") -> None:
        self.root = Path(root)

    def fetch(self, request: MacroDataRequest) -> pd.DataFrame:
        path = self.root / f"{_safe_identifier(request.series_id)}.csv"
        if not path.is_file():
            raise DataProviderError(f"macro-data file not found: {path}")

        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataProviderError(f"unable to read macro-data file: {path}") from exc

        frame = _standardize_columns(frame)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise DataProviderError(f"macro-data file {path} is missing columns: {sorted(missing)}")

        frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce", utc=True).dt.date
        frame["released_at"] = pd.to_datetime(
            frame["released_at"], errors="coerce", utc=True
        ).dt.date
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

        if request.period_start is not None:
            frame = frame[frame["period_end"] >= request.period_start]
        if request.period_end is not None:
            frame = frame[frame["period_end"] <= request.period_end]
        if request.as_of is not None:
            frame = frame[frame["released_at"] <= request.as_of]

        frame = frame[
            (frame["series_id"].astype(str) == request.series_id)
            & (frame["geography"].astype(str) == request.geography)
        ]

        frame = frame[frame["frequency"].astype(str).str.lower() == request.frequency.value]

        return frame.sort_values(["released_at", "period_end"]).reset_index(drop=True)


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized: dict[object, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        normalized[column] = _COLUMN_ALIASES.get(key, key)
    return frame.rename(columns=normalized)


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value.upper())
