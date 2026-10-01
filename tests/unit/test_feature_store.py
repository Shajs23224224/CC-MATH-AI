from datetime import UTC, datetime

import pytest

from core.contracts import FeatureDefinition, FeatureRecord
from core.errors import DataQualityError
from data.feature_store import LocalFeatureStore


def _definition() -> FeatureDefinition:
    return FeatureDefinition(
        feature_id="momentum_20d",
        name="20-day momentum",
        version="1.0.0",
        description="Example point-in-time feature",
        dtype="float64",
        source_columns=("close",),
        frequency="1D",
        lookback=20,
        point_in_time=True,
        methodology="close[t]/close[t-20]-1",
    )


def _record(timestamp: datetime) -> FeatureRecord:
    return FeatureRecord(
        feature_id="momentum_20d",
        feature_version="1.0.0",
        asset_symbol="AAPL",
        timestamp=timestamp,
        value=0.10,
        data_snapshot_id="snapshot-1",
        created_at=timestamp,
    )


def test_feature_store_requires_registered_definition(tmp_path) -> None:
    store = LocalFeatureStore(tmp_path)
    with pytest.raises(DataQualityError):
        store.write((_record(datetime(2026, 1, 1, tzinfo=UTC)),))


def test_feature_store_writes_and_reads_versioned_records(tmp_path) -> None:
    store = LocalFeatureStore(tmp_path)
    store.register_definition(_definition())
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    assert store.write((_record(timestamp),)) == 1
    records = store.read("momentum_20d", "1.0.0", as_of=timestamp)
    assert len(records) == 1
    assert records[0].data_snapshot_id == "snapshot-1"


def test_feature_store_rejects_future_records(tmp_path) -> None:
    store = LocalFeatureStore(tmp_path)
    store.register_definition(_definition())
    record = _record(datetime(2026, 1, 3, tzinfo=timezone.utc))
    with pytest.raises(DataQualityError):
        store.write((record,), as_of=datetime(2026, 1, 2, tzinfo=timezone.utc))


def test_feature_definition_conflicts_are_rejected(tmp_path) -> None:
    store = LocalFeatureStore(tmp_path)
    definition = _definition()
    store.register_definition(definition)
    conflicting = definition.model_copy(update={"methodology": "changed"})
    with pytest.raises(DataQualityError):
        store.register_definition(conflicting)
