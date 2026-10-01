from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AlgorithmMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_schema: str
    output_schema: str
    data_requirements: tuple[str, ...] = ()
    frequency_requirements: tuple[str, ...] = ()
    units: str | None = None
    limitations: tuple[str, ...] = ()


class AlgorithmResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm_id: str
    algorithm_version: str
    timestamp: str
    values: dict[str, float | int | str | bool | None]
    diagnostics: dict[str, Any] = {}
