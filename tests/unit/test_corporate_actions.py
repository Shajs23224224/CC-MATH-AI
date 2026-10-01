from datetime import date

import pandas as pd
import pytest

from core.contracts import (
    Asset,
    AssetType,
    CorporateAction,
    CorporateActionRequest,
    CorporateActionType,
)
from core.errors import DataQualityError, DataProviderError
from data import CorporateActionEngine, apply_split_adjustments, normalize_corporate_actions
from data.providers.corporate_action_csv import LocalCSVCorporateActionProvider


def _asset() -> Asset:
    return Asset(symbol="AAPL", asset_type=AssetType.EQUITY, currency="USD")


def _split() -> CorporateAction:
    return CorporateAction(
        asset=_asset(),
        action_type=CorporateActionType.SPLIT,
        announced_at=date(2026, 5, 1),
        ex_date=date(2026, 5, 10),
        ratio_numerator=2,
        ratio_denominator=1,
        source="test",
    )


def test_split_requires_ratio() -> None:
    with pytest.raises(ValueError):
        CorporateAction(
            asset=_asset(),
            action_type=CorporateActionType.SPLIT,
            ex_date=date(2026, 5, 10),
            source="test",
        )


def test_dividend_requires_cash_amount() -> None:
    with pytest.raises(ValueError):
        CorporateAction(
            asset=_asset(),
            action_type=CorporateActionType.DIVIDEND,
            ex_date=date(2026, 5, 10),
            source="test",
        )


def test_normalize_corporate_actions() -> None:
    frame = pd.DataFrame(
        {
            "action_type": ["split"],
            "announced_at": ["2026-05-01"],
            "ex_date": ["2026-05-10"],
            "ratio_numerator": [2],
            "ratio_denominator": [1],
            "source": ["test"],
        }
    )
    actions = normalize_corporate_actions(frame, _asset())
    assert actions[0].split_factor == 2.0


def test_split_adjusts_only_prior_bars() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-09T00:00:00Z",
                "2026-05-10T00:00:00Z",
            ],
            "open": [100.0, 50.0],
            "high": [102.0, 51.0],
            "low": [99.0, 49.0],
            "close": [101.0, 50.0],
            "volume": [1000.0, 2000.0],
        }
    )
    adjusted = apply_split_adjustments(frame, (_split(),))

    assert adjusted.loc[0, "close"] == 50.5
    assert adjusted.loc[0, "volume"] == 2000.0
    assert adjusted.loc[1, "close"] == 50.0
    assert adjusted.loc[1, "volume"] == 2000.0


def test_reverse_split_increases_prior_prices() -> None:
    action = CorporateAction(
        asset=_asset(),
        action_type=CorporateActionType.REVERSE_SPLIT,
        ex_date=date(2026, 5, 10),
        ratio_numerator=1,
        ratio_denominator=10,
        source="test",
    )
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-05-09T00:00:00Z"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.0],
            "volume": [1000.0],
        }
    )
    adjusted = apply_split_adjustments(frame, (action,))
    assert adjusted.loc[0, "close"] == 100.0
    assert adjusted.loc[0, "volume"] == 100.0


def test_action_engine_requires_file(tmp_path) -> None:
    provider = LocalCSVCorporateActionProvider(tmp_path)
    request = CorporateActionRequest(asset=_asset())
    with pytest.raises(DataProviderError):
        provider.fetch(request)


def test_action_engine_pipeline(tmp_path) -> None:
    path = tmp_path / "AAPL.csv"
    path.write_text(
        "action_type,announced_at,ex_date,ratio_numerator,ratio_denominator,source\n"
        "split,2026-05-01,2026-05-10,2,1,test\n",
        encoding="utf-8",
    )
    request = CorporateActionRequest(asset=_asset(), as_of=date(2026, 5, 2))
    actions, batch = CorporateActionEngine(
        LocalCSVCorporateActionProvider(tmp_path)
    ).fetch(request)

    assert batch.actions_count == 1
    assert actions[0].split_factor == 2.0


def test_as_of_excludes_unannounced_actions(tmp_path) -> None:
    path = tmp_path / "AAPL.csv"
    path.write_text(
        "action_type,ex_date,ratio_numerator,ratio_denominator,source\n"
        "split,2026-05-10,2,1,test\n",
        encoding="utf-8",
    )
    request = CorporateActionRequest(asset=_asset(), as_of=date(2026, 5, 2))
    provider = LocalCSVCorporateActionProvider(tmp_path)
    assert provider.fetch(request).empty
