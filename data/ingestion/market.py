from __future__ import annotations

from core.contracts import MarketDataBatch, MarketDataRequest, MarketBar
from core.errors import DataProviderError

from data.normalization.market import normalize_market_frame
from data.providers.base import MarketDataProvider


class MarketDataEngine:
    """Application service coordinating provider fetch and normalization."""

    def __init__(self, provider: MarketDataProvider) -> None:
        self.provider = provider

    def fetch(
        self,
        request: MarketDataRequest,
    ) -> tuple[tuple[MarketBar, ...], MarketDataBatch]:
        if request.frequency not in self.provider.supported_frequencies:
            raise DataProviderError(
                f"provider {self.provider.name!r} does not support "
                f"frequency {request.frequency.value!r}"
            )

        frame = self.provider.fetch(request)
        bars = normalize_market_frame(frame, request.asset, request.frequency)

        batch = MarketDataBatch(
            request=request,
            bars_count=len(bars),
            first_timestamp=bars[0].timestamp if bars else None,
            last_timestamp=bars[-1].timestamp if bars else None,
        )
        return bars, batch
