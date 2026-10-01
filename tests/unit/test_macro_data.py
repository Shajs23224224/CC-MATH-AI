from datetime import date

import pandas as pd
import pytest

from core.contracts import MacroDataRequest, MacroObservation
from core.errors import DataProviderError, DataQualityError
from data import MacroDataEngine, normalize_macro_frame
from data.providers import LocalCSVMacroProvider


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["CPI", "CPI"],
            "name": ["Inflation", "Inflation"],
            "geography": ["CO", "CO"],
            "period_end": ["2026-01-31", "2026-02-28"],
            "released_at": ["2026-02-10", "2026-03-10"],
            "value": [5.2, 5.0],
            "unit": ["percent", "percent"],
            "frequency": ["monthly", "monthly"],
            "source": ["test", "test"],
        }
    )


def test_macro_observation_rejects_future_period_release() -> None:
    with pytest.raises(ValueError):
        MacroObservation(
            series_id="CPI",
            name="Inflation",
            geography="CO",
            period_end=date(2026, 2, 28),
            released_at=date(2026, 2, 27),
            value=5.0,
            unit="percent",
            frequency="monthly",
            source="test",
        )


def test_macro_request_rejects_inverted_period_window() -> None:
    with pytest.raises(ValueError):
        MacroDataRequest(
            series_id="GDP",
            name="GDP",
            geography="CO",
            period_start=date(2026, 6, 30),
            period_end=date(2026, 3, 31),
            frequency="quarterly",
        )


def test_normalize_macro_frame() -> None:
    observations = normalize_macro_frame(_frame())
    assert len(observations) == 2
    assert observations[0].period_end == date(2026, 1, 31)


def test_normalization_rejects_duplicates() -> None:
    frame = _frame()
    frame.loc[1, "released_at"] = frame.loc[0, "released_at"]
    with pytest.raises(DataQualityError):
        normalize_macro_frame(frame)


def test_local_provider_requires_file(tmp_path) -> None:
    provider = LocalCSVMacroProvider(tmp_path)
    request = MacroDataRequest(
        series_id="CPI",
        name="Inflation",
        geography="CO",
        frequency="monthly",
    )
    with pytest.raises(DataProviderError):
        provider.fetch(request)


def test_local_provider_filters_as_of(tmp_path) -> None:
    path = tmp_path / "CPI.csv"
    _frame().to_csv(path, index=False)
    request = MacroDataRequest(
        series_id="CPI",
        name="Inflation",
        geography="CO",
        frequency="monthly",
        as_of=date(2026, 2, 15),
    )
    observations, _ = MacroDataEngine(LocalCSVMacroProvider(tmp_path)).fetch(request)

    assert len(observations) == 1
    assert observations[0].period_end == date(2026, 1, 31)


def test_macro_engine_pipeline(tmp_path) -> None:
    path = tmp_path / "RATE.csv"
    frame = _frame().copy()
    frame["series_id"] = "RATE"
    frame["name"] = "Policy rate"
    frame["value"] = [9.0, 8.5]
    frame["unit"] = "percent"
    frame.to_csv(path, index=False)

    request = MacroDataRequest(
        series_id="RATE",
        name="Policy rate",
        geography="CO",
        frequency="monthly",
    )
    observations, batch = MacroDataEngine(
        LocalCSVMacroProvider(tmp_path)
    ).fetch(request)

    assert batch.observations_count == len(observations) == 2
    assert observations[0].value == 9.0
