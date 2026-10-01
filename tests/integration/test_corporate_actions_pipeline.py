from datetime import date

from core.contracts import Asset, AssetType, CorporateActionRequest
from data import CorporateActionEngine
from data.providers.corporate_action_csv import LocalCSVCoporateActionProvider


def test_corporate_action_pipeline_is_point_in_time(tmp_path) -> None:
    path = tmp_path / "MSFT.csv"
    path.write_text(
        "action_type,announced_at,ex_date,ratio_numerator,ratio_denominator,source\n"
        "split,2026-06-01,2026-06-15,2,1,test\n"
        "split,2026-08-01,2026-08-15,3,1,test\n",
        encoding="utf-8",
    )

    request = CorporateActionRequest(
        asset=Asset(symbol="MSFT", asset_type=AssetType.EQUITY, currency="USD"),
        as_of=date(2026, 6, 10),
    )
    actions, _ = CorporateActionEngine(LocalCSVCoporateActionProvider(tmp_path)).fetch(request)

    assert len(actions) == 1
    assert actions[0].announced_at == date(2026, 6, 1)
