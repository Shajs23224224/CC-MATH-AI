"""Optional historical market-data adapters (Yahoo Finance, CCXT and Alpha Vantage).

Dependencies are imported lazily so the core package remains usable offline and
without provider-specific packages. API keys must be supplied through environment
variables; never commit credentials.
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from core.contracts import Frequency, MarketDataRequest
from core.errors import DataProviderError

_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_FREQUENCY_MINUTES = {
    Frequency.MINUTE_1: 1,
    Frequency.MINUTE_5: 5,
    Frequency.MINUTE_15: 15,
    Frequency.HOUR_1: 60,
    Frequency.HOUR_4: 240,
    Frequency.DAILY: 1440,
    Frequency.WEEKLY: 10080,
}


def _window(frame: pd.DataFrame, request: MarketDataRequest) -> pd.DataFrame:
    """Apply an inclusive UTC date window and return canonical column names."""
    frame = frame.copy()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    aliases = {"date": "timestamp", "datetime": "timestamp", "adj_close": "close"}
    frame = frame.rename(columns=aliases)
    missing = set(_COLUMNS) - set(frame.columns)
    if missing:
        raise DataProviderError(f"provider returned incomplete OHLCV columns: {sorted(missing)}")
    frame = frame.loc[:, _COLUMNS]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for column in _COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(_COLUMNS)).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    if request.start is not None:
        start = pd.Timestamp(request.start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        frame = frame[frame["timestamp"] >= start]
    if request.end is not None:
        end = pd.Timestamp(request.end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        frame = frame[frame["timestamp"] <= end]
    if frame.empty:
        return pd.DataFrame(columns=list(_COLUMNS))
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise DataProviderError("provider returned invalid OHLCV values")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise DataProviderError("provider returned inconsistent OHLC high values")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise DataProviderError("provider returned inconsistent OHLC low values")
    return frame.reset_index(drop=True)


class YahooFinanceMarketDataProvider:
    """Historical stocks, ETFs, indices, FX and crypto via the optional yfinance package.

    Yahoo Finance availability, symbol coverage and intraday history are subject to
    provider limitations. This is a research adapter, not an exchange-grade feed.
    """

    name = "yahoo_finance"
    supported_frequencies = frozenset(
        {Frequency.MINUTE_1, Frequency.MINUTE_5, Frequency.MINUTE_15,
         Frequency.HOUR_1, Frequency.DAILY, Frequency.WEEKLY}
    )

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        if request.frequency not in self.supported_frequencies:
            raise DataProviderError(f"Yahoo Finance does not support {request.frequency.value}")
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataProviderError(
                'Yahoo Finance adapter requires the optional dependency: '
                'pip install -e ".[market-data]"'
            ) from exc

        intervals = {
            Frequency.MINUTE_1: "1m",
            Frequency.MINUTE_5: "5m",
            Frequency.MINUTE_15: "15m",
            Frequency.HOUR_1: "1h",
            Frequency.DAILY: "1d",
            Frequency.WEEKLY: "1wk",
        }
        kwargs: dict[str, Any] = {
            "tickers": request.asset.symbol,
            "interval": intervals[request.frequency],
            "auto_adjust": request.adjusted,
            "progress": False,
            "threads": False,
        }
        if request.start is not None:
            kwargs["start"] = request.start
        if request.end is not None:
            kwargs["end"] = request.end
        if request.start is None and request.end is None:
            kwargs["period"] = "max"
        try:
            raw = yf.download(**kwargs)
        except Exception as exc:
            raise DataProviderError(f"Yahoo Finance download failed: {exc}") from exc
        if raw.empty:
            return pd.DataFrame(columns=list(_COLUMNS))
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename_axis("timestamp").reset_index()
        raw = raw.rename(
            columns={"Open": "open", "High": "high", "Low": "low",
                     "Close": "close", "Volume": "volume", "Date": "timestamp"}
        )
        return _window(raw, request)


class CCXTHistoricalMarketDataProvider:
    """Historical OHLCV candles from a CCXT-supported crypto exchange (default Binance)."""

    name = "ccxt"
    supported_frequencies = frozenset(
        {Frequency.MINUTE_1, Frequency.MINUTE_5, Frequency.MINUTE_15,
         Frequency.HOUR_1, Frequency.HOUR_4, Frequency.DAILY, Frequency.WEEKLY}
    )

    def __init__(self, exchange_id: str = "binance", page_limit: int = 1000) -> None:
        if not exchange_id.replace("_", "").isalnum():
            raise ValueError("invalid CCXT exchange id")
        if not 1 <= page_limit <= 1000:
            raise ValueError("page_limit must be between 1 and 1000")
        self.exchange_id = exchange_id
        self.page_limit = page_limit

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        if request.frequency not in self.supported_frequencies:
            raise DataProviderError(f"CCXT does not support {request.frequency.value}")
        if request.adjusted:
            raise DataProviderError("CCXT crypto candles are not corporate-action adjusted")
        try:
            import ccxt
        except ImportError as exc:
            raise DataProviderError(
                'CCXT adapter requires the optional dependency: pip install -e ".[market-data]"'
            ) from exc
        intervals = {
            Frequency.MINUTE_1: "1m", Frequency.MINUTE_5: "5m",
            Frequency.MINUTE_15: "15m", Frequency.HOUR_1: "1h",
            Frequency.HOUR_4: "4h", Frequency.DAILY: "1d", Frequency.WEEKLY: "1w",
        }
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({"enableRateLimit": True})
            if not exchange.has.get("fetchOHLCV"):
                raise DataProviderError(f"CCXT exchange {self.exchange_id} has no OHLCV endpoint")
            since = int(pd.Timestamp(request.start).timestamp() * 1000) if request.start else None
            end_ms = int(pd.Timestamp(request.end).timestamp() * 1000) if request.end else None
            rows: list[list[float]] = []
            while True:
                page = exchange.fetch_ohlcv(
                    request.asset.symbol, timeframe=intervals[request.frequency],
                    since=since, limit=self.page_limit,
                )
                if not page:
                    break
                rows.extend(row for row in page if end_ms is None or row[0] <= end_ms)
                next_since = int(page[-1][0]) + _FREQUENCY_MINUTES[request.frequency] * 60_000
                if len(page) < self.page_limit or (since is not None and next_since <= since):
                    break
                if end_ms is not None and next_since > end_ms:
                    break
                since = next_since
            exchange.close()
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(f"CCXT historical download failed: {exc}") from exc
        frame = pd.DataFrame(
            rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
        )
        if frame.empty:
            return pd.DataFrame(columns=list(_COLUMNS))
        frame["timestamp"] = pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True)
        return _window(frame, request)


class AlphaVantageMarketDataProvider:
    """Historical equity/FX/crypto data via Alpha Vantage's JSON API.

    Set ALPHAVANTAGE_API_KEY in the environment. Free-tier limits are imposed by
    Alpha Vantage and may result in a throttling or information message.
    """

    name = "alpha_vantage"
    supported_frequencies = frozenset(
        {Frequency.MINUTE_1, Frequency.MINUTE_5, Frequency.MINUTE_15,
         Frequency.HOUR_1, Frequency.DAILY, Frequency.WEEKLY}
    )

    def __init__(self, api_key: str | None = None, timeout: int = 20) -> None:
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        self.timeout = timeout

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        if not self.api_key:
            raise DataProviderError("set ALPHAVANTAGE_API_KEY; do not store the key in source code")
        params: dict[str, str] = {"apikey": self.api_key, "symbol": request.asset.symbol}
        if request.frequency == Frequency.WEEKLY:
            if request.adjusted:
                params["function"] = "TIME_SERIES_WEEKLY_ADJUSTED"
            else:
                params["function"] = "TIME_SERIES_WEEKLY"
        elif request.frequency == Frequency.DAILY:
            params["function"] = (
                "TIME_SERIES_DAILY_ADJUSTED" if request.adjusted else "TIME_SERIES_DAILY"
            )
            params["outputsize"] = "full"
        elif request.frequency in {
            Frequency.MINUTE_1, Frequency.MINUTE_5, Frequency.MINUTE_15, Frequency.HOUR_1
        }:
            if request.adjusted:
                raise DataProviderError(
                    "Alpha Vantage intraday endpoint does not provide adjusted OHLCV"
                )
            params["function"] = "TIME_SERIES_INTRADAY"
            params["interval"] = {
                Frequency.MINUTE_1: "1min",
                Frequency.MINUTE_5: "5min",
                Frequency.MINUTE_15: "15min",
                Frequency.HOUR_1: "60min",
            }[request.frequency]
            params["outputsize"] = "full"
        else:
            raise DataProviderError(f"Alpha Vantage does not support {request.frequency.value}")

        url = "https://www.alphavantage.co/query?" + urlencode(params)
        try:
            with urlopen(url, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise DataProviderError(f"Alpha Vantage request failed: {exc}") from exc
        if "Error Message" in payload or "Note" in payload or "Information" in payload:
            detail = (
                payload.get("Error Message")
                or payload.get("Note")
                or payload.get("Information")
            )
            raise DataProviderError(f"Alpha Vantage response: {detail}")
        series_key = next(
            (
                key for key, value in payload.items()
                if "Time Series" in key and isinstance(value, dict)
            ),
            None,
        )
        if series_key is None:
            raise DataProviderError("Alpha Vantage response has no time-series payload")
        records: list[dict[str, Any]] = []
        for timestamp, values in payload[series_key].items():
            normalized = {
                str(k).split(". ", 1)[-1].lower().replace(" ", "_"): v
                for k, v in values.items()
            }
            close = float(normalized.get("close", normalized.get("adjusted_close", 0)))
            open_price = float(normalized["open"])
            high = float(normalized["high"])
            low = float(normalized["low"])
            if request.adjusted and "adjusted_close" in normalized and "close" in normalized:
                factor = float(normalized["adjusted_close"]) / float(normalized["close"])
                open_price, high, low, close = (
                    open_price * factor,
                    high * factor,
                    low * factor,
                    float(normalized["adjusted_close"]),
                )
            records.append({
                "timestamp": timestamp, "open": open_price, "high": high, "low": low,
                "close": close, "volume": float(normalized.get("volume", 0)),
            })
        return _window(pd.DataFrame(records, columns=list(_COLUMNS)), request)
