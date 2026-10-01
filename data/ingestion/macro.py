from __future__ import annotations

from core.contracts import MacroDataBatch, MacroDataRequest, MacroObservation
from core.errors import DataProviderError

from data.normalization.macro import normalize_macro_frame
from data.providers.macro_base import MacroDataProvider


class MacroDataEngine:
    """Application service for point-in-time macroeconomic data."""

    def __init__(self, provider: MacroDataProvider) -> None:
        self.provider = provider

    def fetch(
        self,
        request: MacroDataRequest,
    ) -> tuple[tuple[MacroObservation, ...], MacroDataBatch]:
        frame = self.provider.fetch(request)
        observations = normalize_macro_frame(frame)

        filtered = tuple(
            observation
            for observation in observations
            if request.as_of is None or observation.released_at <= request.as_of
        )

        if len(filtered) != len(observations):
            raise DataProviderError("provider returned macro data newer than request.as_of")

        batch = MacroDataBatch(
            request=request,
            observations_count=len(filtered),
            first_period_end=filtered[0].period_end if filtered else None,
            last_period_end=filtered[-1].period_end if filtered else None,
        )
        return filtered, batch
