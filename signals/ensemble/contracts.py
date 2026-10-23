"""Typed contracts for deterministic quantitative signal ensembling."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NormalizationMethod = Literal["bounded", "tanh", "zscore"]
_VERSION_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EnsembleEvidence(BaseModel):
    """One source's timestamped numeric evidence before ensemble aggregation.

    The normalizer is deterministic and bounded in [-1, 1]. Calibration parameters
    must come from training data available at or before the request's as-of time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source_id: str = Field(min_length=1, max_length=160)
    source_version: str = Field(min_length=1, max_length=80)
    observed_at: datetime
    horizon: str = Field(min_length=1, max_length=80)
    raw_score: float
    normalization: NormalizationMethod = "bounded"
    normalization_center: float = 0.0
    normalization_scale: float = Field(default=1.0, gt=0.0, le=1_000_000.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    correlation_group: str | None = Field(default=None, min_length=1, max_length=160)
    eligible: bool = True
    exclusion_reason: str | None = Field(default=None, max_length=500)
    rationale: str = Field(min_length=1, max_length=1000)

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Refuse ambiguous local timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_normalization_domain(self) -> EnsembleEvidence:
        """Ensure bounded inputs cannot silently exceed the declared scale."""
        if self.normalization == "bounded" and not -1.0 <= self.raw_score <= 1.0:
            raise ValueError("bounded raw_score must lie in [-1, 1]")
        if not self.eligible and self.exclusion_reason is not None:
            if not self.exclusion_reason.strip():
                raise ValueError("exclusion_reason must not be blank")
        return self

    @property
    def normalized_score(self) -> float:
        """Return the declared deterministic normalization into [-1, 1]."""
        if self.normalization == "bounded":
            return self.raw_score
        if self.normalization == "tanh":
            return math.tanh(self.raw_score / self.normalization_scale)
        standardized = (self.raw_score - self.normalization_center) / self.normalization_scale
        return math.tanh(standardized)


class SourceWeight(BaseModel):
    """Versioned per-source multiplier; zero disables a source."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source_id: str = Field(min_length=1, max_length=160)
    weight: float = Field(ge=0.0, le=100.0)


class EnsembleConfig(BaseModel):
    """Versioned deterministic aggregation policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    config_id: str = Field(default="quant-ensemble", min_length=1, max_length=160)
    version: str = "1.0.0"
    default_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    source_weights: tuple[SourceWeight, ...] = Field(default=(), max_length=1000)
    min_independent_groups: int = Field(default=2, ge=1, le=1000)
    neutral_threshold: float = Field(default=0.05, ge=0.0, lt=1.0)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Require semantic versioning for reproducible policy references."""
        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("version must use semantic versioning, e.g. 1.0.0")
        return value

    @model_validator(mode="after")
    def validate_source_weights(self) -> EnsembleConfig:
        """Prevent ambiguous overrides for one evidence source."""
        source_ids = [item.source_id for item in self.source_weights]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_weights must not contain duplicate source IDs")
        return self


class EnsembleRequest(BaseModel):
    """All evidence for one asset, horizon, data snapshot and information time."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    as_of: datetime
    horizon: str = Field(min_length=1, max_length=80)
    data_snapshot_id: str = Field(min_length=1, max_length=200)
    evidence: tuple[EnsembleEvidence, ...] = Field(default=(), max_length=1000)

    @field_validator("as_of")
    @classmethod
    def require_as_of_timezone(cls, value: datetime) -> datetime:
        """Refuse request cutoffs without timezone information."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_evidence_scope(self) -> EnsembleRequest:
        """Enforce one source/version per request and forbid future evidence."""
        source_ids = [item.source_id for item in self.evidence]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("each source_id may appear only once per ensemble request")
        for item in self.evidence:
            if item.horizon != self.horizon:
                raise ValueError("all evidence must use the request horizon")
            if item.observed_at > self.as_of:
                raise ValueError("evidence observed_at must not be later than request as_of")
        return self


class EnsembleContribution(BaseModel):
    """Auditable contribution of one accepted evidence source."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source_id: str = Field(min_length=1, max_length=160)
    source_version: str = Field(min_length=1, max_length=80)
    correlation_group: str = Field(min_length=1, max_length=160)
    normalized_score: float = Field(ge=-1.0, le=1.0)
    effective_weight: float = Field(gt=0.0)
    normalized_weight: float = Field(gt=0.0, le=1.0)
    contribution: float = Field(ge=-1.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=1000)
    observed_at: datetime


class ExcludedEvidence(BaseModel):
    """Reason an input did not contribute to the output."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source_id: str = Field(min_length=1, max_length=160)
    source_version: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=500)


class EnsembleResult(BaseModel):
    """Reproducible aggregation output, not a trade or risk authorization."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    as_of: datetime
    horizon: str = Field(min_length=1, max_length=80)
    data_snapshot_id: str = Field(min_length=1, max_length=200)
    config_id: str = Field(min_length=1, max_length=160)
    config_version: str = Field(min_length=1, max_length=80)
    method: Literal["weighted_mean_with_correlation_group_cap"] = (
        "weighted_mean_with_correlation_group_cap"
    )
    score: float = Field(ge=-1.0, le=1.0)
    agreement: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    abstained: bool
    abstention_reason: str | None = Field(default=None, max_length=500)
    source_count: int = Field(ge=0)
    independent_group_count: int = Field(ge=0)
    contributions: tuple[EnsembleContribution, ...] = Field(default=(), max_length=1000)
    excluded_evidence: tuple[ExcludedEvidence, ...] = Field(default=(), max_length=1000)
    input_fingerprint: str

    @field_validator("as_of")
    @classmethod
    def require_result_timezone(cls, value: datetime) -> datetime:
        """Keep result timestamps explicit."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @field_validator("input_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Accept only a hexadecimal SHA-256 fingerprint."""
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("input_fingerprint must be a 64-character lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_aggregation_invariants(self) -> EnsembleResult:
        """Ensure the attribution rows reconcile to the reported score."""
        if self.source_count != len(self.contributions):
            raise ValueError("source_count must match the number of contributions")
        if self.independent_group_count > self.source_count:
            raise ValueError("independent groups cannot exceed contributing source count")
        if self.contributions:
            normalized_total = math.fsum(item.normalized_weight for item in self.contributions)
            contribution_total = math.fsum(item.contribution for item in self.contributions)
            if not math.isclose(normalized_total, 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("normalized contribution weights must sum to one")
            if not math.isclose(contribution_total, self.score, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("score must equal the sum of attributed contributions")
        elif not math.isclose(self.score, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("score must be zero when there are no contributions")
        if self.abstained and not self.abstention_reason:
            raise ValueError("abstained results require an abstention_reason")
        if not self.abstained and self.abstention_reason is not None:
            raise ValueError("non-abstained results must not include an abstention_reason")
        return self


__all__ = [
    "EnsembleConfig",
    "EnsembleContribution",
    "EnsembleEvidence",
    "EnsembleRequest",
    "EnsembleResult",
    "ExcludedEvidence",
    "SourceWeight",
]
