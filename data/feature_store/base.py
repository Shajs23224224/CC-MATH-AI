from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.contracts import FeatureDefinition, FeatureRecord


class FeatureStore(Protocol):
    def register_definition(self, definition: FeatureDefinition) -> None: ...

    def write(
        self,
        records: tuple[FeatureRecord, ...],
        *,
        as_of: datetime | None = None,
    ) -> int: ...

    def read(
        self,
        feature_id: str,
        feature_version: str,
        *,
        asset_symbol: str | None = None,
        as_of: datetime | None = None,
    ) -> tuple[FeatureRecord, ...]: ...
