"""Typed contracts for point-in-time-aware feature engineering."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .quality import FeatureDefinition

FeatureOperation = Literal[
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
]
FeatureCategory = Literal[
    "price_return",
    "volatility",
    "volume",
    "fundamental",
    "macro",
    "cross_sectional",
    "regime",
    "risk",
    "interaction",
    "custom",
]

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
_PERIOD_OPERATIONS = {"lag", "pct_change", "log_return"}


class FeatureTransformSpec(BaseModel):
    """Versioned feature definition plus one allow-listed deterministic transform."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    definition: FeatureDefinition
    operation: FeatureOperation
    category: FeatureCategory = "custom"
    window: int = Field(default=1, ge=1, le=5_000)
    periods: int = Field(default=1, ge=1, le=5_000)
    availability_column: str | None = Field(default=None, min_length=1)
    availability_lag: int = Field(default=0, ge=0, le=250)

    @model_validator(mode="after")
    def validate_sources_and_point_in_time(self) -> FeatureTransformSpec:
        source_count = len(self.definition.source_columns)
        expected_count = 2 if self.operation in _TWO_SOURCE_OPERATIONS else 1
        if source_count != expected_count:
            raise ValueError(f"{self.operation} requires exactly {expected_count} source column(s)")
        if len(set(self.definition.source_columns)) != source_count:
            raise ValueError("source_columns must be unique")
        if any(not name.strip() for name in self.definition.source_columns):
            raise ValueError("source_columns must not be blank")
        if self.definition.lookback < self.required_lookback:
            raise ValueError(
                "FeatureDefinition.lookback must cover the transform's required lookback"
            )
        if (
            self.category in _EXTERNAL_CATEGORIES
            and self.availability_column is None
            and self.availability_lag < 1
        ):
            raise ValueError(
                "fundamental, macro, regime and risk features require availability_column "
                "or availability_lag >= 1"
            )
        return self

    @property
    def required_lookback(self) -> int:
        """Rows required before a feature can be fully populated."""
        if self.operation in _WINDOW_OPERATIONS:
            return self.window
        if self.operation in _PERIOD_OPERATIONS:
            return self.periods
        return 0


class FeatureEngineeringConfig(BaseModel):
    """Resource and temporal controls for the stateless feature pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_column: str = Field(default="timestamp", min_length=1)
    asset_column: str | None = Field(default=None, min_length=1)
    max_rows: int = Field(default=50_000, ge=12, le=250_000)
    max_features: int = Field(default=256, ge=1, le=1_000)
    require_point_in_time: bool = True


class FeatureEngineeringReport(BaseModel):
    """Auditable lineage summary returned alongside the generated feature frame."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    pipeline_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    schema_version: Literal["1.0"] = "1.0"
    numpy_version: str = Field(min_length=1)
    pandas_version: str = Field(min_length=1)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_rows: int = Field(ge=1)
    output_rows: int = Field(ge=1)
    input_columns: tuple[str, ...] = Field(min_length=1)
    generated_feature_ids: tuple[str, ...] = Field(min_length=1)
    feature_definitions: tuple[FeatureDefinition, ...] = Field(min_length=1)
    maximum_lookback: int = Field(ge=0)
    rows_with_warmup_missing_values: int = Field(ge=0)
    point_in_time_safe: bool
    chronology_verified: Literal[True] = True
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report_alignment(self) -> FeatureEngineeringReport:
        if self.input_rows != self.output_rows:
            raise ValueError("feature engineering must preserve row count")
        if len(self.generated_feature_ids) != len(self.feature_definitions):
            raise ValueError("feature identifiers and definitions must align")
        if len(set(self.generated_feature_ids)) != len(self.generated_feature_ids):
            raise ValueError("generated feature identifiers must be unique")
        if self.rows_with_warmup_missing_values > self.output_rows:
            raise ValueError("warm-up missing row count exceeds output row count")
        return self


__all__ = [
    "FeatureCategory",
    "FeatureEngineeringConfig",
    "FeatureEngineeringReport",
    "FeatureOperation",
    "FeatureTransformSpec",
]
