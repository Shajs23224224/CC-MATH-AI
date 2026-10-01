from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

import pandas as pd

from core.contracts import Frequency, MarketDataRequest


class MarketDataProvider(Protocol):
    """Port implemented by every market-data adapter."""

    @property
    def name(self) -> str: ...

    @property
    def supported_frequencies(self) -> frozenset[Frequency]: ...

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame: ...


class ProviderCapabilities:
    def __init__(
        self,
        provider_name: str,
        supported_frequencies: frozenset[Frequency],
        earliest: datetime | None = None,
        latest: datetime | None = None,
        extra: Mapping[str, str] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.supported_frequencies = supported_frequencies
        self.earliest = earliest
        self.latest = latest
        self.extra = dict(extra or {})
