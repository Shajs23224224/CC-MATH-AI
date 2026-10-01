from __future__ import annotations

from core.contracts import FundamentalDataBatch, FundamentalDataRequest, FundamentalSnapshot
from core.errors import DataProviderError
from data.normalization.fundamentals import normalize_fundamental_frame
from data.providers.fundamental_base import FundamentalDataProvider


class FundamentalDataEngine:
    """Application service for point-in-time fundamental data."""

    def __init__(self, provider: FundamentalDataProvider) -> None:
        self.provider = provider

    def fetch(
        self,
        request: FundamentalDataRequest,
    ) -> tuple[tuple[FundamentalSnapshot, ...], FundamentalDataBatch]:
        frame = self.provider.fetch(request)
        snapshots = normalize_fundamental_frame(frame, request.asset)

        if request.as_of is not None and any(
            snapshot.reported_at > request.as_of for snapshot in snapshots
        ):
            raise DataProviderError("provider returned data newer than request.as_of")

        batch = FundamentalDataBatch(
            request=request,
            snapshots_count=len(snapshots),
            first_period_end=snapshots[0].period_end if snapshots else None,
            last_period_end=snapshots[-1].period_end if snapshots else None,
        )
        return snapshots, batch
