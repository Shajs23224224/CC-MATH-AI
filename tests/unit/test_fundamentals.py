from datetime import date

import pandas as pd
import pytest

from core.contracts import (
    Asset,
    AssetType,
    FundamentalDataRequest,
    FundamentalSnapshot,
)
from core.errors import DataProviderError, DataQualityError
from data import FundamentalDataEngine, normalize_fundamental_frame
from data.providers import LocalCSVFundamentalProvider


def _asset() -> Asset:
    return Asset(symbol="AAPL", asset_type=AssetType.EQUITY, currency="USD")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_end": ["2026-03-31", "2026-06-30"],
            "reported_at": ["2026-04-25", "2026-07-28"],
            "revenue": [1000, 1200],
            "ebitda": [250, 300],
            "ebit": [200, 240],
            "net_income": [150, 180],
            "eps": [1.5, 1.8],
            "free_cash_flow": [130, 170],
            "capex": [40, 45],
            "cash": [600, 650],
            "debt": [300, 280],
            "equity": [900, 950],
            "dividends": [20, 20],
            "shares_outstanding": [100, 100],
        }
    )


def test_fundamental_snapshot_rejects_lookahead_date() -> None:
    with pytest.raises(ValueError):
        FundamentalSnapshot(
            asset=_asset(),
            period_end=date(2026, 6, 30),
            reported_at=date(2026, 6, 29),
        )


def test_normalize_fundamental_frame() -> None:
    snapshots = normalize_fundamental_frame(_frame(), _asset())
    assert len(snapshots) == 2
    assert snapshots[0].period_end == date(2026, 3, 31)
    assert snapshots[1].reported_at == date(2026, 7, 28)


def test_normalization_rejects_duplicate_snapshot() -> None:
    frame = _frame()
    frame.loc[1, "period_end"] = frame.loc[0, "period_end"]
    frame.loc[1, "reported_at"] = frame.loc[0, "reported_at"]
    with pytest.raises(DataQualityError):
        normalize_fundamental_frame(frame, _asset())


def test_engine_respects_as_of() -> None:
    frame = _frame()
    filtered = frame[frame["reported_at"] <= "2026-04-30"]
    snapshots = normalize_fundamental_frame(filtered, _asset())
    assert len(snapshots) == 1
    assert snapshots[0].reported_at == date(2026, 4, 25)


def test_local_provider_requires_file(tmp_path) -> None:
    provider = LocalCSVFundamentalProvider(tmp_path)
    request = FundamentalDataRequest(asset=_asset())
    with pytest.raises(DataProviderError):
        provider.fetch(request)


def test_local_provider_filters_as_of(tmp_path) -> None:
    path = tmp_path / "AAPL.csv"
    _frame().to_csv(path, index=False)

    provider = LocalCSVFundamentalProvider(tmp_path)
    request = FundamentalDataRequest(asset=_asset(), as_of=date(2026, 4, 30))
    frame = provider.fetch(request)

    assert len(frame) == 1
    assert frame.iloc[0]["reported_at"] == date(2026, 4, 25)


def test_fundamental_engine_pipeline(tmp_path) -> None:
    path = tmp_path / "MSFT.csv"
    _frame().to_csv(path, index=False)
    request = FundamentalDataRequest(
        asset=Asset(symbol="MSFT", asset_type=AssetType.EQUITY, currency="USD"),
        as_of=date(2026, 12, 31),
    )

    snapshots, batch = FundamentalDataEngine(LocalCSVFundamentalProvider(tmp_path)).fetch(request)

    assert batch.snapshots_count == len(snapshots) == 2
    assert batch.first_period_end == date(2026, 3, 31)
