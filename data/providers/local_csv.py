from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from core.contracts import Frequency, MarketDataRequest
from core.errors import DataProviderError

from .base import MarketDataProvider


_REQUIRED_COLUMNS = frozenset({"timestamp", "open", "high", "low", "close", "volume"})
_COLUMN_ALIASES = {
    "datetime": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "adj_close": "close",
}


class LocalCSVMarketDataProvider:
    """Local development provider using one CSV file per asset."""

    name = "local_csv"
    supported_frequencies = frozenset(
        frequency for frequency in Frequency if frequency != Frequency.TICK
    )

    def __init__(self, root: str | Path = "data/local/market") -> None:
        self.root = Path(root)

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        if request.adjusted:
            raise DataProviderError(
                "local_csv does not provide corporate-action-adjusted prices; use F10"
            )

        path = self.root / f"{_safe_symbol(request.asset.symbol)}.csv"
        if not path.is_file():
            raise DataProviderError(f"market-data file not found: {path}")

        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataProviderError(f"unable to read market-data file: {path}") from exc

        frame = _standardize_columns(frame)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise DataProviderError(
                f"market-data file {path} is missing columns: {sorted(missing)}"
            )

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if request.start is not None:
            start = pd.Timestamp(request.start)
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            frame = frame[frame["timestamp"] >= start]
        if request.end is not None:
            end = pd.Timestamp(request.end)
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            frame = frame[frame["timestamp"] <= end]

        return frame.reset_index(drop=True)


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_")
        normalized[column] = _COLUMN_ALIASES.get(key, key)
    return frame.rename(columns=normalized)


def _safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.upper())
