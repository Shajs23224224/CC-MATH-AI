"""Public F24 cointegration and statistical-arbitrage API."""

from .models import (
    adf_test,
    benjamini_hochberg,
    engle_granger_test,
    fit_cointegrating_pair,
    johansen_test,
    kpss_test,
    phillips_perron_test,
    spread_zscore,
    unit_root_diagnostics,
)

__all__ = [
    "adf_test",
    "benjamini_hochberg",
    "engle_granger_test",
    "fit_cointegrating_pair",
    "johansen_test",
    "kpss_test",
    "phillips_perron_test",
    "spread_zscore",
    "unit_root_diagnostics",
]
