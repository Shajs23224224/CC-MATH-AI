from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.contracts import Frequency, MarketDataRequest
from core.errors import DataProviderError

_FREQUENCIES = {
    Frequency.MINUTE_1: "m1",
    Frequency.MINUTE_5: "m5",
    Frequency.MINUTE_15: "m15",
    Frequency.HOUR_1: "h1",
    Frequency.HOUR_4: "h4",
    Frequency.DAILY: "d1",
}
_INSTRUMENTS = {
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "AUDUSD": "audusd",
    "USDCAD": "usdcad",
    "USDCHF": "usdchf",
    "NZDUSD": "nzdusd",
    "XAUUSD": "xauusd",
    "XAGUSD": "xagusd",
}
_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class DukascopyMarketDataProvider:
    """Historical OHLCV adapter using the free dukascopy-node CLI.

    Requires Node.js/npm and downloads only when a matching local cache file is
    absent. Data is stored under data/cache by default, which is ignored by Git.
    """

    name = "dukascopy"

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/dukascopy",
        command: tuple[str, ...] = ("npx", "--yes", "dukascopy-node"),
        timeout_seconds: int = 900,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.cache_dir = Path(cache_dir)
        self.command = command
        self.timeout_seconds = timeout_seconds

    @property
    def supported_frequencies(self) -> frozenset[Frequency]:
        return frozenset(_FREQUENCIES)

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        if request.adjusted:
            raise DataProviderError(
                "Dukascopy provides unadjusted market prices; adjusted=True is unsupported"
            )
        instrument = _INSTRUMENTS.get(request.asset.symbol.replace("/", "").upper())
        if instrument is None:
            raise DataProviderError(
                f"unsupported Dukascopy instrument: {request.asset.symbol!r}; "
                f"supported symbols: {', '.join(sorted(_INSTRUMENTS))}"
            )
        timeframe = _FREQUENCIES.get(request.frequency)
        if timeframe is None:
            raise DataProviderError(
                f"Dukascopy does not support requested frequency {request.frequency.value!r}"
            )
        if request.start is None or request.end is None:
            raise DataProviderError(
                "Dukascopy requests require both start and end dates to bound downloads"
            )

        start = _utc_timestamp(request.start)
        end = _utc_timestamp(request.end)
        if end <= start:
            raise DataProviderError("end must be after start")
        cache_path = self.cache_dir / (
            f"{instrument}_{timeframe}_{start:%Y%m%dT%H%M%S}_"
            f"{end:%Y%m%dT%H%M%S}.csv"
        )
        if cache_path.is_file():
            return _read_market_csv(cache_path)

        with TemporaryDirectory(prefix="ccmath-dukascopy-") as temporary_dir:
            output_dir = Path(temporary_dir)
            command = [
                *self.command,
                "--instrument", instrument,
                "--date-from", start.strftime("%Y-%m-%d"),
                "--date-to", (end.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "--timeframe", timeframe,
                "--format", "csv",
                "--directory", str(output_dir),
                "--volumes",
                "--silent",
                "--retries", "2",
                "--retry-pause", "1000",
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DataProviderError(
                    "Unable to execute dukascopy-node. Install Node.js/npm and verify "
                    "network access; see https://www.dukascopy-node.app/"
                ) from exc

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
                raise DataProviderError(
                    f"Dukascopy download failed (exit {completed.returncode}): {detail[-1200:]}"
                )

            csv_files = sorted(output_dir.rglob("*.csv"))
            if not csv_files:
                detail = (completed.stderr or completed.stdout or "no CSV file produced").strip()
                raise DataProviderError(f"Dukascopy returned no CSV data: {detail[-1200:]}")
            frames = []
            for path in csv_files:
                try:
                    frames.append(pd.read_csv(path))
                except (OSError, pd.errors.ParserError) as exc:
                    raise DataProviderError(f"Unable to parse Dukascopy output: {path.name}") from exc

        frame = pd.concat(frames, ignore_index=True)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing = set(_COLUMNS) - set(frame.columns)
        if missing:
            raise DataProviderError(f"Dukascopy CSV missing columns: {sorted(missing)}")

        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame["timestamp"] = pd.to_datetime(timestamps, unit="ms", utc=True, errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[:, list(_COLUMNS)].dropna(subset=list(_COLUMNS))
        frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        if frame.empty:
            raise DataProviderError(
                f"Dukascopy returned no usable bars for {request.asset.symbol} "
                f"({request.frequency.value}) in the requested window"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_cache = cache_path.with_suffix(".tmp")
        frame.to_csv(temporary_cache, index=False)
        temporary_cache.replace(cache_path)
        return frame.reset_index(drop=True)


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _read_market_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, parse_dates=["timestamp"])
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise DataProviderError(f"Unable to read cached Dukascopy data: {path}") from exc
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    missing = set(_COLUMNS) - set(frame.columns)
    if missing or frame["timestamp"].isna().any():
        raise DataProviderError(f"Invalid cached Dukascopy file: {path}")
    return frame.loc[:, _COLUMNS].sort_values("timestamp").reset_index(drop=True)
