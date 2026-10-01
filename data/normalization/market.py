from __future__ import annotations

from typing import cast

import pandas as pd

from core.contracts import Asset, Frequency, MarketBar
from core.errors import DataQualityError

from .common import normalize_market_frequency, normalize_timestamp_utc


def normalize_market_frame(
    frame: pd.DataFrame,
    asset: Asset,
    frequency: Frequency | str,
) -> tuple[MarketBar, ...]:
    """Normalize OHLCV data into immutable canonical MarketBar contracts."""
    required = ("timestamp", "open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataQualityError(f"missing normalized columns: {missing}")

    canonical_frequency = normalize_market_frequency(frequency)
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
            timestamp=normalize_timestamp_utc(row.timestamp),
            open=cast(float, row.open),
            high=cast(float, row.high),
            low=cast(float, row.low),
            close=cast(float, row.close),
            volume=cast(float, row.volume),
        )
        bar.validate_ohlc()
        bars.append(bar)

    if bars and canonical_frequency == Frequency.TICK:
        raise DataQualityError("OHLCV normalization does not support tick data")

    return tuple(bars)
