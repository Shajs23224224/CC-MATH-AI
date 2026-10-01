from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from .enums import AssetType, DataQualityStatus


class Asset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    asset_type: AssetType
    currency: str = Field(min_length=3, max_length=3)
    exchange: str | None = None

    @field_validator("symbol", "currency")
    @classmethod
    def normalize_upper(cls, value: str) -> str:
        return value.strip().upper()


class MarketBar(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> "MarketBar":
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open and close")
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class PriceSeries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    timestamps: tuple[datetime, ...]
    values: tuple[float, ...]
    frequency: str
    data_quality: DataQualityStatus = DataQualityStatus.UNKNOWN

    @model_validator(mode="after")
    def validate_alignment(self) -> "PriceSeries":
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have equal length")
        if any(value <= 0 for value in self.values):
            raise ValueError("price values must be positive")
        if any(left >= right for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")
        return self
