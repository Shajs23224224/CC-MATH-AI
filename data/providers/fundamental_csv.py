from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from core.contracts import FundamentalDataRequest
from core.errors import DataProviderError



_REQUIRED_COLUMNS = frozenset({"period_end", "reported_at"})
_COLUMN_ALIASES = {
    "period": "period_end",
    "periodend": "period_end",
    "period_end_date": "period_end",
    "report_date": "reported_at",
    "reported_date": "reported_at",
    "filing_date": "reported_at",
    "announcement_date": "reported_at",
}


class LocalCSVFundamentalProvider:
    """Local CSV fundamental provider for development and point-in-time testing."""

    name = "local_csv_fundamentals"

    def __init__(self, root: str | Path = "data/local/fundamentals") -> None:
        self.root = Path(root)

    def fetch(self, request: FundamentalDataRequest) -> pd.DataFrame:
        path = self.root / f"{_safe_symbol(request.asset.symbol)}.csv"
        if not path.is_file():
            raise DataProviderError(f"fundamental-data file not found: {path}")

        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataProviderError(
                f"unable to read fundamental-data file: {path}"
            ) from exc

        frame = _standardize_columns(frame)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise DataProviderError(
                f"fundamental-data file {path} is missing columns: {sorted(missing)}"
            )

        frame["period_end"] = pd.to_datetime(
            frame["period_end"], errors="coerce", utc=True
        ).dt.date
        frame["reported_at"] = pd.to_datetime(
            frame["reported_at"], errors="coerce", utc=True
        ).dt.date

        if request.period_start is not None:
            frame = frame[frame["period_end"] >= request.period_start]
        if request.period_end is not None:
            frame = frame[frame["period_end"] <= request.period_end]
        if request.as_of is not None:
            frame = frame[frame["reported_at"] <= request.as_of]

        frame = frame.sort_values(["reported_at", "period_end"]).reset_index(drop=True)
        if request.limit is not None:
            frame = frame.tail(request.limit).reset_index(drop=True)

        return frame


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized: dict[object, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        normalized[column] = _COLUMN_ALIASES.get(key, key)
    return frame.rename(columns=normalized)


def _safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.upper())
