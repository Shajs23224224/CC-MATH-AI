from __future__ import annotations

from core.contracts import CorporateAction, CorporateActionBatch, CorporateActionRequest
from data.normalization.corporate_actions import normalize_corporate_actions
from data.providers.corporate_action_base import CorporateActionProvider


class CorporateActionEngine:
    """Application service for corporate-action ingestion and normalization."""

    def __init__(self, provider: CorporateActionProvider) -> None:
        self.provider = provider

    def fetch(
        self,
        request: CorporateActionRequest,
    ) -> tuple[tuple[CorporateAction, ...], CorporateActionBatch]:
        frame = self.provider.fetch(request)
        actions = normalize_corporate_actions(frame, request.asset)

        if request.as_of is not None:
            future_announcements = tuple(
                action
                for action in actions
                if action.announced_at is not None
                and action.announced_at > request.as_of
            )
            if future_announcements:
                raise ValueError("provider returned corporate actions newer than request.as_of")

        batch = CorporateActionBatch(
            request=request,
            actions_count=len(actions),
        )
        return actions, batch
