from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from core.contracts import Asset, Frequency, MarketBar
from core.errors import DataQualityError


def normalize_market_frame(
    frame: pd.DataFrame,
    asset: Asset,
    frequency: Frequency,
) -> tuple[MarketBar, ...]:
    """Normalize an OHLCV frame into immutable MarketBar contracts."""

    required = ("timestamp", "open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataQualityError(f"missing normalized columns: {missing}")

    work = frame.loc[:, required].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)

    numeric = ("open", "high", "low", "close", "volume")
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    if work["timestamp"].isna().any():
        raise DataQualityError("market data contains invalid timestamps")
    if work["timestamp"].duplicated().any():
        raise DataQualityError("market data contains duplicate timestamps")
    if work.loc[:, numeric].isna().any().any():
        raise DataQualityError("market data contains non-numeric or missing OHLCV values")

    work = work.sort_values("timestamp").reset_index(drop=True)

    bars: list[MarketBar] = []
    for row in work.itertuples(index=False):
        bar = MarketBar(
            asset=asset,
            timestamp=_as_datetime(row.timestamp),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        bars.append(bar)
        bar.validate_ohlc()

    if bars and frequency == Frequency.TICK:
        raise DataQualityError("OHLCV normalization does not support tick data")

    return tuple(bars)


def _as_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.to_pydatetime().astimezone(timezone.utc)
