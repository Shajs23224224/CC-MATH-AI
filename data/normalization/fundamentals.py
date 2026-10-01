from __future__ import annotations

import pandas as pd

from core.contracts import Asset, FundamentalSnapshot
from core.errors import DataQualityError
from .common import normalize_dataframe_columns, normalize_date


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
    work = normalize_dataframe_columns(frame)
    missing = [column for column in ("period_end", "reported_at") if column not in work.columns]
    if missing:
        raise DataQualityError(f"missing fundamental columns: {missing}")

    work["period_end"] = pd.to_datetime(work["period_end"], errors="coerce", utc=True).dt.date
    work["reported_at"] = pd.to_datetime(work["reported_at"], errors="coerce", utc=True).dt.date

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
        values: dict[str, float | None] = {}
        for field in _FIELDS:
            if hasattr(row, field):
                value = getattr(row, field)
                values[field] = None if pd.isna(value) else float(value)

        snapshots.append(
            FundamentalSnapshot(
                asset=asset,
                period_end=normalize_date(row.period_end),
                reported_at=normalize_date(row.reported_at),
                **values,
            )
        )

    return tuple(snapshots)
