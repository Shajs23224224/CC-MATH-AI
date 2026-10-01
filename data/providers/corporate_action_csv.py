from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from core.contracts import CorporateActionRequest
from core.errors import DataProviderError

_REQUIRED_COLUMNS = frozenset({"action_type", "source"})
_COLUMN_ALIASES = {
    "type": "action_type",
    "event_type": "action_type",
    "announcement_date": "announced_at",
    "announcement": "announced_at",
    "ex": "ex_date",
    "record": "record_date",
    "pay_date": "payable_date",
    "payment_date": "payable_date",
    "new_ticker": "new_symbol",
    "new_symbol": "new_symbol",
    "ratio_num": "ratio_numerator",
    "ratio_den": "ratio_denominator",
    "amount": "cash_amount",
    "dividend_amount": "cash_amount",
    "event_id": "source_event_id",
}


class LocalCSVCorporateActionProvider:
    """Local CSV corporate-action provider for development and testing."""

    name = "local_csv_corporate_actions"

    def __init__(self, root: str | Path = "data/local/corporate_actions") -> None:
        self.root = Path(root)

    def fetch(self, request: CorporateActionRequest) -> pd.DataFrame:
        path = self.root / f"{_safe_symbol(request.asset.symbol)}.csv"
        if not path.is_file():
            raise DataProviderError(f"corporate-action file not found: {path}")

        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataProviderError(
                f"unable to read corporate-action file: {path}"
            ) from exc

        frame = _standardize_columns(frame)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise DataProviderError(
                f"corporate-action file {path} is missing columns: {sorted(missing)}"
            )

        for column in ("announced_at", "ex_date", "record_date", "payable_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column], errors="coerce", utc=True
                ).dt.date

        for column in ("ratio_numerator", "ratio_denominator", "cash_amount"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if request.start is not None and "ex_date" in frame.columns:
            frame = frame[frame["ex_date"] >= request.start]
        if request.end is not None and "ex_date" in frame.columns:
            frame = frame[frame["ex_date"] <= request.end]

        if request.as_of is not None:
            if "announced_at" not in frame.columns:
                return frame.iloc[0:0].copy()
            frame = frame[
                frame["announced_at"].notna()
                & (frame["announced_at"] <= request.as_of)
            ]

        return frame.reset_index(drop=True)


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized: dict[object, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        normalized[column] = _COLUMN_ALIASES.get(key, key)
    return frame.rename(columns=normalized)


def _safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.upper())
