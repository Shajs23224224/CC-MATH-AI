from datetime import UTC, datetime

import pandas as pd

from core.contracts import FeatureDefinition, FeatureRecord
from data.feature_store import LocalFeatureStore
from data.quality import DataQualityEngine


def test_quality_gate_then_feature_store(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "close": [100.0, 101.0],
        }
    )

    cutoff = datetime(2026, 1, 2, tzinfo=UTC)
    report = DataQualityEngine().check_frame(
        frame,
        dataset_id="AAPL-prices",
        required_columns=("timestamp", "close"),
        key_columns=("timestamp",),
        numeric_columns=("close",),
        timestamp_column="timestamp",
        as_of=cutoff,
    )
    assert report.is_usable is True

    store = LocalFeatureStore(tmp_path)
    store.register_definition(
        FeatureDefinition(
            feature_id="daily_return",
            name="Daily Return",
            version="1.0.0",
            description="Close-to-close simple return",
            dtype="float64",
            source_columns=("close",),
            frequency="1D",
            lookback=1,
            point_in_time=True,
            methodology="close[t]/close[t-1]-1",
        )
    )

    store.write(
        (
            FeatureRecord(
                feature_id="daily_return",
                feature_version="1.0.0",
                asset_symbol="AAPL",
                timestamp=cutoff,
                value=0.01,
                data_snapshot_id="AAPL-prices-2026-01-02",
                created_at=cutoff,
            ),
        ),
        as_of=cutoff,
    )

    records = store.read("daily_return", "1.0.0", as_of=cutoff)
    assert len(records) == 1
