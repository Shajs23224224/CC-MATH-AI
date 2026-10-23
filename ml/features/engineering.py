"""Deterministic, bounded and point-in-time-conscious feature engineering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from core.contracts.feature_engineering import (
    FeatureEngineeringConfig,
    FeatureEngineeringReport,
    FeatureOperation,
    FeatureTransformSpec,
)
from ml.errors import MLInputError, MLTrainingError

_EXTERNAL_CATEGORIES = {"fundamental", "macro", "regime", "risk"}
_TWO_SOURCE_OPERATIONS = {"interaction", "ratio"}
_WINDOW_OPERATIONS = {
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_sum",
    "rolling_zscore",
    "ewm_mean",
}
_SUPPORTED_OPERATIONS: tuple[FeatureOperation, ...] = (
    "identity",
    "lag",
    "pct_change",
    "log_return",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_sum",
    "rolling_zscore",
    "ewm_mean",
    "interaction",
    "ratio",
    "cross_sectional_rank",
    "cross_sectional_zscore",
)


def supported_feature_operations() -> tuple[FeatureOperation, ...]:
    """Return the exact allow-list of transforms supported by F32."""
    return _SUPPORTED_OPERATIONS


class FeatureEngineer:
    """Generate trailing-window features from already point-in-time-aligned source data.

    Rows are never sorted or dropped implicitly. Warm-up and unavailable values remain NaN
    so downstream training code can apply an explicit missing-data policy.
    """

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()

    def transform(
        self,
        frame: pd.DataFrame,
        specs: Sequence[FeatureTransformSpec],
        *,
        dataset_id: str,
    ) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
        """Return a copied frame with features and a reproducible lineage report."""
        if not dataset_id.strip():
            raise MLInputError("dataset_id must not be blank")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise MLInputError("frame must be a non-empty pandas DataFrame")
        if len(frame) > self.config.max_rows:
            raise MLInputError("input rows exceed the configured feature-engineering budget")
        if not frame.columns.is_unique:
            raise MLInputError("input column names must be unique")
        if not specs:
            raise MLInputError("at least one feature specification is required")
        if len(specs) > self.config.max_features:
            raise MLInputError("feature count exceeds the configured budget")
        if len({spec.definition.feature_id for spec in specs}) != len(specs):
            raise MLInputError("feature_id values must be unique within a pipeline")

        timestamp_column = self.config.timestamp_column
        if timestamp_column not in frame.columns:
            raise MLInputError(f"missing timestamp column: {timestamp_column}")
        asset_column = self.config.asset_column
        if asset_column is not None and asset_column not in frame.columns:
            raise MLInputError(f"missing asset column: {asset_column}")

        result = frame.copy(deep=True)
        try:
            timestamps = pd.to_datetime(result[timestamp_column], errors="raise", utc=True)
        except (ValueError, TypeError, OverflowError) as exc:
            raise MLInputError("timestamps must be valid date/time values") from exc
        if timestamps.isna().any():
            raise MLInputError("timestamps must not be missing")
        try:
            ordered = timestamps.is_monotonic_increasing
        except (TypeError, ValueError) as exc:
            raise MLInputError("timestamps must use a consistent timezone") from exc
        if not ordered:
            raise MLInputError("rows must already be ordered by non-decreasing timestamp")
        result[timestamp_column] = timestamps

        if asset_column is not None:
            if result[asset_column].isna().any():
                raise MLInputError("asset identifiers must not be missing")
            if result.duplicated([timestamp_column, asset_column]).any():
                raise MLInputError("each timestamp/asset pair must be unique")
        elif timestamps.duplicated().any():
            raise MLInputError("timestamps must be unique when asset_column is not configured")

        output_names = [spec.definition.feature_id for spec in specs]
        collisions = set(output_names).intersection(result.columns)
        if collisions:
            raise MLInputError(
                f"generated features would overwrite input columns: {sorted(collisions)}"
            )

        point_in_time_safe = True
        for spec in specs:
            definition = spec.definition
            if self.config.require_point_in_time and not definition.point_in_time:
                raise MLInputError(
                    f"feature {definition.feature_id} is not declared point-in-time safe"
                )
            if (
                spec.category in _EXTERNAL_CATEGORIES
                and spec.availability_column is None
                and spec.availability_lag < 1
            ):
                raise MLInputError(
                    f"feature {definition.feature_id} lacks point-in-time availability controls"
                )
            if not definition.point_in_time:
                point_in_time_safe = False
            if (
                spec.operation in {"cross_sectional_rank", "cross_sectional_zscore"}
                and asset_column is None
            ):
                raise MLInputError(
                    f"feature {definition.feature_id} requires asset_column for cross-sectional transforms"
                )
            for source_column in definition.source_columns:
                if source_column not in result.columns:
                    raise MLInputError(f"missing source column: {source_column}")
            if spec.availability_column is not None:
                if spec.availability_column not in result.columns:
                    raise MLInputError(
                        f"missing availability column: {spec.availability_column}"
                    )
                available_at = pd.to_datetime(
                    result[spec.availability_column], errors="coerce", utc=True
                )
                if available_at.isna().all():
                    raise MLInputError(
                        f"availability column {spec.availability_column} has no valid values"
                    )

            generated = self._calculate(result, spec, timestamps)
            generated = self._apply_availability_lag(generated, result, spec, asset_column)
            array = generated.to_numpy(dtype=np.float64, na_value=np.nan)
            if np.isinf(array).any():
                raise MLTrainingError(
                    f"feature {definition.feature_id} produced infinite values"
                )
            result[definition.feature_id] = array

        features = result[output_names]
        output_fingerprint = _fingerprint(features)
        input_fingerprint = _fingerprint(frame)
        warmup_missing_rows = int(features.isna().any(axis=1).sum())
        maximum_lookback = max(
            max(spec.required_lookback, spec.availability_lag) for spec in specs
        )
        pipeline_identity = {
            "dataset_id": dataset_id,
            "input_fingerprint": input_fingerprint,
            "feature_definitions": [
                {
                    "definition": spec.definition.model_dump(mode="json"),
                    "operation": spec.operation,
                    "category": spec.category,
                    "window": spec.window,
                    "periods": spec.periods,
                    "availability_column": spec.availability_column,
                    "availability_lag": spec.availability_lag,
                }
                for spec in specs
            ],
        }
        pipeline_id = hashlib.sha256(
            json.dumps(
                pipeline_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:24]
        report = FeatureEngineeringReport(
            pipeline_id=pipeline_id,
            dataset_id=dataset_id,
            numpy_version=np.__version__,
            pandas_version=pd.__version__,
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            input_rows=len(frame),
            output_rows=len(result),
            input_columns=tuple(str(column) for column in frame.columns),
            generated_feature_ids=tuple(output_names),
            feature_definitions=tuple(spec.definition for spec in specs),
            maximum_lookback=maximum_lookback,
            rows_with_warmup_missing_values=warmup_missing_rows,
            point_in_time_safe=point_in_time_safe,
            limitations=(
                "Source rows must be aligned to the information actually available at each timestamp.",
                "Cross-sectional ranks depend on a point-in-time eligible universe supplied by the caller.",
                "Warm-up and unavailable values remain NaN; downstream missing-data handling is explicit.",
                "Generated features are inputs, not causal explanations, trading signals or guarantees.",
            ),
        )
        return result, report

    def _calculate(
        self,
        frame: pd.DataFrame,
        spec: FeatureTransformSpec,
        timestamps: pd.Series,
    ) -> pd.Series:
        definition = spec.definition
        sources = [
            self._source_series(frame, column, spec, timestamps)
            for column in definition.source_columns
        ]
        operation = spec.operation
        first = sources[0]

        if operation in {"identity", "lag", "pct_change", "log_return"}:
            def univariate(values: pd.Series) -> pd.Series:
                if operation == "identity":
                    return values.copy()
                if operation == "lag":
                    return values.shift(spec.periods)
                if operation == "pct_change":
                    return values.pct_change(periods=spec.periods, fill_method=None)
                if (values.dropna() <= 0.0).any():
                    raise MLInputError(
                        f"log_return feature {definition.feature_id} requires positive values"
                    )
                return np.log(values / values.shift(spec.periods))

            return self._transform_by_asset(first, frame, univariate)

        if operation in _WINDOW_OPERATIONS:
            def rolling(values: pd.Series) -> pd.Series:
                if operation == "ewm_mean":
                    return values.ewm(
                        span=spec.window,
                        adjust=False,
                        min_periods=spec.window,
                    ).mean()
                window = values.rolling(window=spec.window, min_periods=spec.window)
                if operation == "rolling_mean":
                    return window.mean()
                if operation == "rolling_std":
                    return window.std(ddof=0)
                if operation == "rolling_min":
                    return window.min()
                if operation == "rolling_max":
                    return window.max()
                if operation == "rolling_sum":
                    return window.sum()
                mean = window.mean()
                standard_deviation = window.std(ddof=0).replace(0.0, np.nan)
                return (values - mean) / standard_deviation

            return self._transform_by_asset(first, frame, rolling)

        if operation in _TWO_SOURCE_OPERATIONS:
            second = sources[1]
            if operation == "interaction":
                return first * second
            return first.div(second.where(second != 0.0))

        if operation == "cross_sectional_rank":
            counts = first.groupby(timestamps, sort=False).transform("count")
            ranks = first.groupby(timestamps, sort=False).rank(method="average", pct=True)
            return ranks.where(counts >= 2)
        if operation == "cross_sectional_zscore":
            means = first.groupby(timestamps, sort=False).transform("mean")
            variance = (first - means).pow(2).groupby(timestamps, sort=False).transform("mean")
            standard_deviation = np.sqrt(variance).replace(0.0, np.nan)
            return (first - means).div(standard_deviation)

        raise MLInputError(f"unsupported feature operation: {operation}")

    def _source_series(
        self,
        frame: pd.DataFrame,
        column: str,
        spec: FeatureTransformSpec,
        timestamps: pd.Series,
    ) -> pd.Series:
        try:
            values = pd.to_numeric(frame[column], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise MLInputError(f"source column {column} must be numeric") from exc
        if np.isinf(values.to_numpy(dtype=np.float64, na_value=np.nan)).any():
            raise MLInputError(f"source column {column} contains infinite values")
        if spec.availability_column is None:
            return values
        try:
            availability = pd.to_datetime(
                frame[spec.availability_column], errors="coerce", utc=True
            )
            known = availability.notna() & (availability <= timestamps)
        except (TypeError, ValueError) as exc:
            raise MLInputError("could not compare availability times with row timestamps") from exc
        masked = values.where(known)
        asset_column = self.config.asset_column
        if asset_column is not None:
            return masked.groupby(frame[asset_column], sort=False).ffill()
        return masked.ffill()

    def _transform_by_asset(
        self,
        values: pd.Series,
        frame: pd.DataFrame,
        operation: Callable[[pd.Series], pd.Series],
    ) -> pd.Series:
        asset_column = self.config.asset_column
        if asset_column is None:
            return operation(values)
        return values.groupby(frame[asset_column], sort=False, group_keys=False).transform(
            operation
        )

    def _apply_availability_lag(
        self,
        values: pd.Series,
        frame: pd.DataFrame,
        spec: FeatureTransformSpec,
        asset_column: str | None,
    ) -> pd.Series:
        if spec.availability_lag == 0:
            return values
        if asset_column is None:
            return values.shift(spec.availability_lag)
        return values.groupby(frame[asset_column], sort=False, group_keys=False).shift(
            spec.availability_lag
        )

def _fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered rows, column names and dtypes without serializing model objects."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


__all__ = ["FeatureEngineer", "supported_feature_operations"]
