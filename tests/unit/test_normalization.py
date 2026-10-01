from datetime import UTC, datetime

import pandas as pd
import pytest

from core.contracts import Frequency, MacroFrequency, NormalizationReport, NormalizationStatus
from core.errors import DataQualityError
from data.normalization.common import (
    normalize_currency_code,
    normalize_dataframe_columns,
    normalize_identifier,
    normalize_macro_frequency,
    normalize_market_frequency,
    normalize_symbol,
    normalize_timestamp_utc,
    normalize_unit,
    preserve_value_without_conversion,
)


def test_symbol_currency_and_identifier_are_canonicalized() -> None:
    assert normalize_symbol("  aapl ") == "AAPL"
    assert normalize_currency_code(" usd ") == "USD"
    assert normalize_identifier(" cpi ") == "CPI"


def test_invalid_currency_is_rejected() -> None:
    with pytest.raises(DataQualityError):
        normalize_currency_code("US")


def test_timestamp_is_utc() -> None:
    value = normalize_timestamp_utc("2026-10-01T12:00:00-05:00")
    assert value == datetime(2026, 10, 1, 17, 0, tzinfo=UTC)


def test_frequency_aliases_are_canonical() -> None:
    assert normalize_market_frequency("daily") == Frequency.DAILY
    assert normalize_market_frequency("1h") == Frequency.HOUR_1
    assert normalize_macro_frequency("Q") == MacroFrequency.QUARTERLY
    assert normalize_macro_frequency("monthly") == MacroFrequency.MONTHLY


def test_units_are_canonicalized_without_numeric_conversion() -> None:
    assert normalize_unit("%") == "percent"
    value = 123.45
    assert preserve_value_without_conversion(value) == value


def test_dataframe_column_collisions_are_rejected() -> None:
    frame = pd.DataFrame({"Date": [1], "date": [2]})
    with pytest.raises(DataQualityError):
        normalize_dataframe_columns(frame, {"date": "timestamp"})


def test_normalization_report_is_typed() -> None:
    report = NormalizationReport(
        domain="market",
        input_rows=10,
        output_rows=10,
        status=NormalizationStatus.NORMALIZED,
        currency_conversions=0,
        unit_conversions=0,
    )
    assert report.timezone == "UTC"
