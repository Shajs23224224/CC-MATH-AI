from datetime import date

from core.contracts import MacroDataRequest
from data import MacroDataEngine
from data.providers import LocalCSVMacroProvider


def test_macro_pipeline_is_point_in_time(tmp_path) -> None:
    path = tmp_path / "GDP.csv"
    path.write_text(
        "series_id,name,geography,period_end,released_at,value,unit,frequency,source\n"
        "GDP,GDP,CO,2026-03-31,2026-05-15,100,local_currency,quarterly,test\n"
        "GDP,GDP,CO,2026-06-30,2026-08-15,102,local_currency,quarterly,test\n",
        encoding="utf-8",
    )

    request = MacroDataRequest(
        series_id="GDP",
        name="GDP",
        geography="CO",
        frequency="quarterly",
        as_of=date(2026, 6, 1),
    )
    observations, batch = MacroDataEngine(LocalCSVMacroProvider(tmp_path)).fetch(request)

    assert batch.observations_count == 1
    assert observations[0].value == 100.0
