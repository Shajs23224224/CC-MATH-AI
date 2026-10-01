from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TechnicalIndicatorSeries(BaseModel):
    """Immutable, timestamp-aligned technical indicator output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    timestamps: tuple[datetime, ...]
    values: tuple[float | None, ...]

    @model_validator(mode="after")
    def validate_alignment(self) -> TechnicalIndicatorSeries:
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have equal length")
        if any(
            left >= right
            for left, right in zip(self.timestamps[:-1], self.timestamps[1:], strict=True)
        ):
            raise ValueError("timestamps must be strictly increasing")
        return self
