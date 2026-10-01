from __future__ import annotations

from datetime import date, datetime, UTC
from typing import TypeVar

import pandas as pd

from core.contracts import Frequency, MacroFrequency
from core.errors import DataQualityError

_FREQUENCY_ALIASES = {
    "tick": Frequency.TICK,
    "1t": Frequency.TICK,
    "1m": Frequency.MINUTE_1,
    "1min": Frequency.MINUTE_1,
    "minute": Frequency.MINUTE_1,
    "minute_1": Frequency.MINUTE_1,
    "5m": Frequency.MINUTE_5,
    "5min": Frequency.MINUTE_5,
    "15m": Frequency.MINUTE_15,
    "15min": Frequency.MINUTE_15,
    "1h": Frequency.HOUR_1,
    "60m": Frequency.HOUR_1,
    "hourly": Frequency.HOUR_1,
    "4h": Frequency.HOUR_4,
    "1d": Frequency.DAILY,
    "d": Frequency.DAILY,
    "daily": Frequency.DAILY,
    "1w": Frequency.WEEKLY,
    "w": Frequency.WEEKLY,
    "weekly": Frequency.WEEKLY,
}

_MACRO_FREQUENCY_ALIASES = {
    "daily": MacroFrequency.DAILY,
    "d": MacroFrequency.DAILY,
    "weekly": MacroFrequency.WEEKLY,
    "w": MacroFrequency.WEEKLY,
    "monthly": MacroFrequency.MONTHLY,
    "m": MacroFrequency.MONTHLY,
    "quarterly": MacroFrequency.QUARTERLY,
    "q": MacroFrequency.QUARTERLY,
    "annual": MacroFrequency.ANNUAL,
    "annually": MacroFrequency.ANNUAL,
    "yearly": MacroFrequency.ANNUAL,
    "y": MacroFrequency.ANNUAL,
}

_UNIT_ALIASES = {
    "%": "percent",
    "pct": "percent",
    "percentage": "percent",
    "percent": "percent",
    "usd": "USD",
    "eur": "EUR",
    "gbp": "GBP",
    "cop": "COP",
    "jpy": "JPY",
    "cny": "CNY",
    "local_currency": "LOCAL_CURRENCY",
}


def normalize_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if not value:
        raise DataQualityError("symbol cannot be empty")
    return value


def normalize_currency_code(currency: str) -> str:
    value = currency.strip().upper()
    if len(value) != 3 or not value.isalpha():
        raise DataQualityError(f"invalid ISO-style currency code: {currency!r}")
    return value


def normalize_identifier(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise DataQualityError("identifier cannot be empty")
    return normalized


def normalize_unit(unit: str) -> str:
    value = unit.strip()
    if not value:
        raise DataQualityError("unit cannot be empty")
    return _UNIT_ALIASES.get(value.lower(), value)


def normalize_market_frequency(value: str | Frequency) -> Frequency:
    if isinstance(value, Frequency):
        return value
    key = value.strip().lower()
    try:
        return _FREQUENCY_ALIASES[key]
    except KeyError as exc:
        raise DataQualityError(f"unsupported market frequency: {value!r}") from exc


def normalize_macro_frequency(value: str | MacroFrequency) -> MacroFrequency:
    if isinstance(value, MacroFrequency):
        return value
    key = value.strip().lower()
    try:
        return _MACRO_FREQUENCY_ALIASES[key]
    except KeyError as exc:
        raise DataQualityError(f"unsupported macro frequency: {value!r}") from exc


def normalize_timestamp_utc(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise DataQualityError("invalid timestamp")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.to_pydatetime().astimezone(UTC)


def normalize_date(value: object) -> date:
    timestamp = normalize_timestamp_utc(value)
    return timestamp.date()


def normalize_dataframe_columns(
    frame: pd.DataFrame,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    alias_map = aliases or {}
    normalized: dict[object, str] = {}

    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        normalized[column] = alias_map.get(key, key)

    renamed = frame.rename(columns=normalized)
    duplicates = renamed.columns[renamed.columns.duplicated()].tolist()
    if duplicates:
        raise DataQualityError(f"normalization created duplicate columns: {duplicates}")
    return renamed


T = TypeVar("T")


def preserve_value_without_conversion(value: T) -> T:
    """Document the F11 rule: normalization must not imply a numeric conversion."""
    return value
