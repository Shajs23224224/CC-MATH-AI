"""Typed metadata contracts for the F33 algorithm registry."""
from __future__ import annotations
import re
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, field_validator

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")

class AlgorithmLifecycle(StrEnum):
    """Lifecycle state; availability is not a claim of predictive validity."""
    EXPERIMENTAL = "experimental"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"

class AlgorithmTestStatus(StrEnum):
    """Latest recorded test evidence state."""
    NOT_RUN = "not_run"
    PASSING = "passing"
    FAILING = "failing"

class AlgorithmPort(BaseModel):
    """One named algorithm input or output."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    dtype: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    unit: str | None = Field(default=None, max_length=100)
    required: bool = True

class AlgorithmParameter(BaseModel):
    """A parameter declaration, not an executable schema."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    dtype: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    default: str | int | float | bool | None = None
    minimum: float | None = None
    maximum: float | None = None

    @field_validator("maximum")
    @classmethod
    def validate_bounds(cls, value: float | None, info: object) -> float | None:
        minimum = getattr(info, "data", {}).get("minimum")
        if value is not None and minimum is not None and value < minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        return value

class AlgorithmMetadata(BaseModel):
    """Immutable, versioned metadata used for lookup and compatibility checks."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    algorithm_id: str = Field(min_length=3, max_length=160)
    name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=40)
    summary: str = Field(min_length=1, max_length=1000)
    method: str = Field(min_length=1, max_length=2000)
    inputs: tuple[AlgorithmPort, ...] = ()
    outputs: tuple[AlgorithmPort, ...] = Field(min_length=1)
    parameters: tuple[AlgorithmParameter, ...] = ()
    units: tuple[str, ...] = ()
    frequencies: tuple[str, ...] = ()
    data_requirements: tuple[str, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)
    tags: tuple[str, ...] = ()
    implementation_ref: str | None = Field(default=None, max_length=300)
    lifecycle: AlgorithmLifecycle = AlgorithmLifecycle.EXPERIMENTAL
    test_status: AlgorithmTestStatus = AlgorithmTestStatus.NOT_RUN

    @field_validator("algorithm_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError("algorithm_id must be a lowercase slug")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("version must use semantic versioning, e.g. 1.0.0")
        return value

    @field_validator("inputs", "outputs", "parameters")
    @classmethod
    def validate_unique_names(cls, value: tuple[BaseModel, ...]) -> tuple[BaseModel, ...]:
        names = [getattr(item, "name") for item in value]
        if len(names) != len(set(names)):
            raise ValueError("port and parameter names must be unique within their group")
        return value
