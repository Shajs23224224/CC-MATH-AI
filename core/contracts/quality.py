from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .enums import DataQualityStatus


class DataQualitySeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1)
    severity: DataQualitySeverity
    message: str = Field(min_length=1)
    field: str | None = None
    count: int = Field(default=1, ge=1)


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    checked_at: datetime
    rows: int = Field(ge=0)
    accepted_rows: int = Field(ge=0)
    rejected_rows: int = Field(ge=0)
    status: DataQualityStatus
    issues: tuple[DataQualityIssue, ...] = ()
    leakage_detected: bool = False
    survivorship_bias_risk: bool = False

    @property
    def is_usable(self) -> bool:
        return self.status != DataQualityStatus.INVALID and not self.leakage_detected


class FeatureDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    source_columns: tuple[str, ...] = ()
    frequency: str = Field(min_length=1)
    lookback: int = Field(default=0, ge=0)
    point_in_time: bool = True
    methodology: str = Field(min_length=1)


class FeatureRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_id: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    asset_symbol: str = Field(min_length=1)
    timestamp: datetime
    value: float
    data_snapshot_id: str = Field(min_length=1)
    created_at: datetime


class FeatureSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_set_id: str = Field(min_length=1)
    feature_definitions: tuple[str, ...] = ()
    data_snapshot_id: str = Field(min_length=1)
    records: tuple[FeatureRecord, ...] = ()
    created_at: datetime
