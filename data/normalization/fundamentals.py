from __future__ import annotations

from datetime import date

import pandas as pd

from core.contracts import Asset, FundamentalSnapshot
from core.errors import DataQualityError


_FIELDS = (
    "revenue",
    "ebitda",
    "ebit",
    "net_income",
    "eps",
    "free_cash_flow",
    "capex",
    "cash",
    "debt",
    "equity",
    "dividends",
    "shares_outstanding",
)


def normalize_fundamental_frame(
    frame: pd.DataFrame,
    asset: Asset,
) -> tuple[FundamentalSnapshot, ...]:
    required = ("period_end", "reported_at")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataQualityError(f"missing fundamental columns: {missing}")

    work = frame.copy()
    work["period_end"] = _parse_dates(work["period_end"], "period_end")
    work["reported_at"] = _parse_dates(work["reported_at"], "reported_at")

    for field in _FIELDS:
        if field in work.columns:
            work[field] = pd.to_numeric(work[field], errors="coerce")

    if work["period_end"].isna().any() or work["reported_at"].isna().any():
        raise DataQualityError("fundamental data contains invalid dates")

    if work.duplicated(subset=["period_end", "reported_at"]).any():
        raise DataQualityError("duplicate fundamental snapshots detected")

    work = work.sort_values(["reported_at", "period_end"]).reset_index(drop=True)

    snapshots: list[FundamentalSnapshot] = []
    for row in work.itertuples(index=False):
        values = {}
        for field in _FIELDS:
            if hasattr(row, field):
                value = getattr(row, field)
                values[field] = None if pd.isna(value) else float(value)

        snapshot = FundamentalSnapshot(
            asset=asset,
            period_end=_as_date(row.period_end),
            reported_at=_as_date(row.reported_at),
            **values,
        )
        snapshots.append(snapshot)

    return tuple(snapshots)


def _parse_dates(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dt.date
    parsed.name = name
    return parsed


def _as_date(value: object) -> date:
    if not isinstance(value, date):
        raise DataQualityError("normalized fundamental date has invalid type")
    return value
