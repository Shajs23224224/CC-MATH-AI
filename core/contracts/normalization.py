from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class NormalizationStatus(StrEnum):
    NORMALIZED = "normalized"
    WARNING = "warning"
    REJECTED = "rejected"


class NormalizationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = Field(min_length=1)
    input_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)
    status: NormalizationStatus
    timezone: str = "UTC"
    currency_conversions: int = Field(ge=0)
    unit_conversions: int = Field(ge=0)
    notes: tuple[str, ...] = ()
