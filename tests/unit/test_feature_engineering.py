"""Behavioral tests for F32 feature definitions, temporal safety and lineage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.contracts.feature_engineering import (
    FeatureEngineeringConfig,
    FeatureTransformSpec,
)
from core.contracts.quality import FeatureDefinition
from ml.errors import MLInputError
from ml.features import FeatureEngineer, supported_feature_operations


def _spec(
    feature_id: str,
    operation: str,
    sources: tuple[str, ...],
    *,
    lookback: int = 0,
    window: int = 1,
    periods: int = 1,
    category: str = "custom",
    availability_column: str | None = None,
    availability_lag: int = 0,
) -> FeatureTransformSpec:
    return FeatureTransformSpec(
        definition=FeatureDefinition(
            feature_id=feature_id,
            name=feature_id.replace("_", " ").title(),
            version="1.0.0",
            description="Synthetic F32 unit fixture",
            dtype="float64",
            source_columns=sources,
            frequency="1D",
            lookback=lookback,
            point_in_time=True,
            methodology=f"{operation}({', '.join(sources)})",
        ),
        operation=operation,
        category=category,
        window=window,
        periods=periods,
        availability_column=availability_column,
        availability_lag=availability_lag,
    )


def _frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="D"),
            "close": np.arange(100.0, 100.0 + n),
            "volume": np.arange(1.0, n + 1.0),
        }
    )


def test_operation_registry_is_explicit_and_stable() -> None:
    operations = supported_feature_operations()
    assert "rolling_mean" in operations
    assert "cross_sectional_rank" in operations
    assert len(operations) == len(set(operations))


def test_trailing_rolling_and_return_features_preserve_rows_and_warmup() -> None:
    frame = _frame()
    specs = (
        _spec("close_return_1d", "pct_change", ("close",), lookback=1),
        _spec("close_mean_5d", "rolling_mean", ("close",), lookback=5, window=5),
    )
    result, report = FeatureEngineer().transform(frame, specs, dataset_id="synthetic-price-v1")

    assert len(result) == len(frame)
    assert np.isnan(result.loc[0, "close_return_1d"])
    assert result.loc[1, "close_return_1d"] == pytest.approx(0.01)
    assert np.isnan(result.loc[3, "close_mean_5d"])
    assert result.loc[4, "close_mean_5d"] == pytest.approx(102.0)
    assert report.generated_feature_ids == ("close_return_1d", "close_mean_5d")
    assert report.maximum_lookback == 5
    assert report.point_in_time_safe is True
    assert len(report.input_fingerprint) == 64
    assert len(report.output_fingerprint) == 64


def test_future_rows_cannot_change_prior_trailing_features() -> None:
    first = _frame()
    altered = first.copy(deep=True)
    altered.loc[25:, "close"] = altered.loc[25:, "close"] * 100.0
    spec = _spec("mean_5", "rolling_mean", ("close",), lookback=5, window=5)

    before, _ = FeatureEngineer().transform(first, (spec,), dataset_id="fixture")
    after, _ = FeatureEngineer().transform(altered, (spec,), dataset_id="fixture")

    pd.testing.assert_series_equal(before.loc[:24, "mean_5"], after.loc[:24, "mean_5"])


def test_lag_and_log_return_are_trailing_only() -> None:
    frame = _frame()
    specs = (
        _spec("close_lag_2", "lag", ("close",), lookback=2, periods=2),
        _spec("log_return_1", "log_return", ("close",), lookback=1),
    )
    result, _ = FeatureEngineer().transform(frame, specs, dataset_id="prices")
    assert result.loc[2, "close_lag_2"] == pytest.approx(100.0)
    assert result.loc[1, "log_return_1"] == pytest.approx(np.log(101.0 / 100.0))


def test_zero_denominator_ratio_returns_missing_not_infinity() -> None:
    frame = _frame()
    frame["volume"] = 0.0
    spec = _spec("price_volume_ratio", "ratio", ("close", "volume"))
    result, _ = FeatureEngineer().transform(frame, (spec,), dataset_id="zero-volume")
    assert result["price_volume_ratio"].isna().all()


def test_cross_sectional_features_require_panel_and_are_timestamp_scoped() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]
            ),
            "asset_symbol": ["A", "B", "A", "B"],
            "close": [10.0, 20.0, 30.0, 10.0],
        }
    )
    engineer = FeatureEngineer(FeatureEngineeringConfig(asset_column="asset_symbol"))
    spec = _spec(
        "cross_rank",
        "cross_sectional_rank",
        ("close",),
        category="cross_sectional",
    )
    result, _ = engineer.transform(frame, (spec,), dataset_id="cross-section")
    assert result.loc[0, "cross_rank"] == pytest.approx(0.5)
    assert result.loc[1, "cross_rank"] == pytest.approx(1.0)
    assert result.loc[2, "cross_rank"] == pytest.approx(1.0)
    assert result.loc[3, "cross_rank"] == pytest.approx(0.5)


def test_per_asset_rolling_never_crosses_asset_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]
            ),
            "asset_symbol": ["A", "B", "A", "B"],
            "close": [10.0, 100.0, 20.0, 200.0],
        }
    )
    spec = _spec("mean_2", "rolling_mean", ("close",), lookback=2, window=2)
    result, _ = FeatureEngineer(
        FeatureEngineeringConfig(asset_column="asset_symbol")
    ).transform(frame, (spec,), dataset_id="panel")
    assert np.isnan(result.loc[0, "mean_2"])
    assert np.isnan(result.loc[1, "mean_2"])
    assert result.loc[2, "mean_2"] == pytest.approx(15.0)
    assert result.loc[3, "mean_2"] == pytest.approx(150.0)


def test_macro_feature_is_carried_forward_only_after_release() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="D"),
            "macro_value": [1.0, 2.0, 2.0, 3.0],
            "available_at": pd.to_datetime(
                ["2025-01-01", "2025-01-03", "2025-01-03", "2025-01-04"]
            ),
        }
    )
    spec = _spec(
        "macro_level",
        "identity",
        ("macro_value",),
        category="macro",
        availability_column="available_at",
    )
    result, report = FeatureEngineer().transform(frame, (spec,), dataset_id="macro")
    assert result["macro_level"].tolist() == [1.0, 1.0, 2.0, 3.0]
    assert report.point_in_time_safe is True


def test_external_feature_without_availability_controls_is_rejected() -> None:
    with pytest.raises(ValueError, match="availability_column"):
        _spec("cpi", "identity", ("macro_value",), category="macro")


def test_duplicate_timestamps_and_unsorted_rows_are_rejected() -> None:
    frame = _frame()
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    spec = _spec("level", "identity", ("close",))
    with pytest.raises(MLInputError, match="unique"):
        FeatureEngineer().transform(frame, (spec,), dataset_id="duplicates")

    frame = _frame()
    frame.loc[5, "timestamp"] = pd.Timestamp("2024-01-01")
    with pytest.raises(MLInputError, match="ordered"):
        FeatureEngineer().transform(frame, (spec,), dataset_id="unsorted")


def test_non_finite_values_and_feature_name_collisions_fail_explicitly() -> None:
    frame = _frame()
    frame.loc[4, "close"] = np.inf
    spec = _spec("feature", "identity", ("close",))
    with pytest.raises(MLInputError, match="infinite"):
        FeatureEngineer().transform(frame, (spec,), dataset_id="infinite")

    frame = _frame()
    collision = _spec("close", "identity", ("volume",))
    with pytest.raises(MLInputError, match="overwrite"):
        FeatureEngineer().transform(frame, (collision,), dataset_id="collision")


def test_resource_budgets_and_non_pit_definitions_are_enforced() -> None:
    frame = _frame()
    spec = _spec("level", "identity", ("close",))
    engineer = FeatureEngineer(FeatureEngineeringConfig(max_rows=20))
    with pytest.raises(MLInputError, match="budget"):
        engineer.transform(frame, (spec,), dataset_id="over-budget")

    unsafe_definition = spec.definition.model_copy(update={"point_in_time": False})
    unsafe = spec.model_copy(update={"definition": unsafe_definition})
    with pytest.raises(MLInputError, match="not declared point-in-time"):
        FeatureEngineer().transform(frame, (unsafe,), dataset_id="non-pit")


def test_feature_spec_enforces_declared_lookback() -> None:
    with pytest.raises(ValueError, match="lookback"):
        _spec("mean", "rolling_mean", ("close",), lookback=2, window=5)
