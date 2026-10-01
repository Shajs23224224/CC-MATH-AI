"""Data ingestion, normalization and quality boundaries."""

from .ingestion import FundamentalDataEngine, MacroDataEngine, MarketDataEngine
from .normalization import (
    normalize_fundamental_frame,
    normalize_macro_frame,
    normalize_market_frame,
)
from .providers import (
    FundamentalDataProvider,
    LocalCSVFundamentalProvider,
    LocalCSVMacroProvider,
    LocalCSVMarketDataProvider,
    MacroDataProvider,
    MarketDataProvider,
)
from .quality import MarketDataQualitySummary

__all__ = [
    "FundamentalDataEngine",
    "FundamentalDataProvider",
    "LocalCSVFundamentalProvider",
    "LocalCSVMacroProvider",
    "LocalCSVMarketDataProvider",
    "MacroDataEngine",
    "MacroDataProvider",
    "MarketDataEngine",
    "MarketDataProvider",
    "MarketDataQualitySummary",
    "normalize_fundamental_frame",
    "normalize_macro_frame",
    "normalize_market_frame",
]
