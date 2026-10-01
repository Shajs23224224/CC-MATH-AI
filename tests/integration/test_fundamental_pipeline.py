from datetime import date

from core.contracts import Asset, AssetType, FundamentalDataRequest
from data import FundamentalDataEngine
from data.providers import LocalCSVFundamentalProvider


def test_point_in_time_fundamental_pipeline(tmp_path) -> None:
    path = tmp_path / "IBM.csv"
    path.write_text(
        "period_end,reported_at,revenue,ebitda,eps,free_cash_flow\n"
        "2026-03-31,2026-04-25,1000,250,1.5,130\n"
        "2026-06-30,2026-07-28,1200,300,1.8,170\n",
        encoding="utf-8",
    )

    request = FundamentalDataRequest(
        asset=Asset(symbol="IBM", asset_type=AssetType.EQUITY, currency="USD"),
        as_of=date(2026, 5, 1),
    )

    snapshots, batch = FundamentalDataEngine(LocalCSVFundamentalProvider(tmp_path)).fetch(request)

    assert batch.snapshots_count == 1
    assert snapshots[0].period_end == date(2026, 3, 31)
    assert snapshots[0].reported_at == date(2026, 4, 25)
