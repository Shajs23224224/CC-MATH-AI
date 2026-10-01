from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.contracts import FeatureDefinition, FeatureRecord
from core.errors import DataQualityError


class LocalFeatureStore:
    """Append-only JSONL feature store for development and reproducible tests."""

    def __init__(self, root: str | Path = "data/local/features") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._definitions = self.root / "definitions.jsonl"
        self._records = self.root / "records.jsonl"

    def register_definition(self, definition: FeatureDefinition) -> None:
        existing = self._load_definitions()
        key = (definition.feature_id, definition.version)
        if key in existing:
            if existing[key] != definition:
                raise DataQualityError(
                    f"feature definition conflict: {definition.feature_id}@{definition.version}"
                )
            return

        with self._definitions.open("a", encoding="utf-8") as handle:
            handle.write(definition.model_dump_json() + "\n")

    def write(
        self,
        records: tuple[FeatureRecord, ...],
        *,
        as_of: datetime | None = None,
    ) -> int:
        if not records:
            return 0

        if as_of is not None:
            cutoff = _as_utc(as_of)
            if any(record.timestamp > cutoff for record in records):
                raise DataQualityError("feature store rejected records newer than as_of")

        definitions = self._load_definitions()
        for record in records:
            if (record.feature_id, record.feature_version) not in definitions:
                raise DataQualityError(
                    f"unregistered feature: {record.feature_id}@{record.feature_version}"
                )

        with self._records.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
        return len(records)

    def read(
        self,
        feature_id: str,
        feature_version: str,
        *,
        asset_symbol: str | None = None,
        as_of: datetime | None = None,
    ) -> tuple[FeatureRecord, ...]:
        if not self._records.exists():
            return ()

        cutoff = _as_utc(as_of) if as_of is not None else None
        result: list[FeatureRecord] = []

        for line in self._records.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = FeatureRecord.model_validate_json(line)
            if record.feature_id != feature_id or record.feature_version != feature_version:
                continue
            if asset_symbol is not None and record.asset_symbol != asset_symbol:
                continue
            if cutoff is not None and _as_utc(record.timestamp) > cutoff:
                continue
            result.append(record)

        return tuple(sorted(result, key=lambda item: (item.asset_symbol, item.timestamp)))

    def _load_definitions(self) -> dict[tuple[str, str], FeatureDefinition]:
        if not self._definitions.exists():
            return {}

        result: dict[tuple[str, str], FeatureDefinition] = {}
        for line in self._definitions.read_text(encoding="utf-8").splitlines():
            if line:
                definition = FeatureDefinition.model_validate_json(line)
                result[(definition.feature_id, definition.version)] = definition
        return result


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
