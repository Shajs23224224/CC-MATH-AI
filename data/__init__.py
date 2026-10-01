"""Data ingestion, normalization and quality boundaries."""

from .ingestion import FundamentalDataEngine, MarketDataEngine
from .normalization import normalize_fundamental_frame, normalize_market_frame
from .providers import (
    FundamentalDataProvider,
    LocalCSVFundamentalProvider,
    LocalCSVMarketDataProvider,
    MarketDataProvider,
)
from .quality import MarketDataQualitySummary

__all__ = [
    "FundamentalDataEngine",
    "FundamentalDataProvider",
    "LocalCSVFundamentalProvider",
    "LocalCSVMarketDataProvider",
    "MarketDataEngine",
    "MarketDataProvider",
    "MarketDataQualitySummary",
    "normalize_fundamental_frame",
    "normalize_market_frame",
]
