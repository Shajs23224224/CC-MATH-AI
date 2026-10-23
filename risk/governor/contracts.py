"""Typed F38 Portfolio Risk Governor contracts."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.contracts import RiskAction, RiskMetric
from portfolio.sizing.contracts import SizingResult as F37SizingResult

_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

GovernorPositionStatus = Literal["APPROVED", "REDUCED", "ZERO_SIZE", "BLOCKED"]


class RiskGovernorConfig(BaseModel):
    """Versioned final risk policy; hard controls block and soft controls reduce."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    config_id: str = Field(default="portfolio-risk-governor", min_length=1, max_length=160)
    version: str = "1.0.0"
    max_position_weight: float = Field(default=0.25, gt=0.0, le=1.0)
    max_gross_leverage: float = Field(default=1.0, gt=0.0, le=10.0)
    max_concentration_hhi: float = Field(default=0.60, gt=0.0, le=1.0)
    max_sector_weight: float = Field(default=0.60, gt=0.0, le=1.0)
    max_abs_factor_exposure: float = Field(default=0.75, gt=0.0, le=10.0)
    max_abs_correlation: float = Field(default=0.98, gt=0.0, le=1.0)
    max_drawdown_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    max_daily_loss_fraction: float = Field(default=0.03, gt=0.0, le=1.0)
    max_liquidation_days: float = Field(default=5.0, gt=0.0)
    max_tail_loss_fraction: float = Field(default=0.10, gt=0.0, le=10.0)
    max_turnover_ratio: float = Field(default=1.0, gt=0.0, le=10.0)
    min_data_quality_score: float = Field(default=0.90, ge=0.0, le=1.0)
    min_model_quality_score: float = Field(default=0.85, ge=0.0, le=1.0)
    max_liquidity_participation: float = Field(default=0.10, gt=0.0, le=1.0)
    reduction_factor: float = Field(default=0.50, gt=0.0, lt=1.0)
    max_input_age_seconds: int = Field(default=300, ge=1, le=86400)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Require a semantic version for reproducible policy replay."""
        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("version must use semantic versioning, e.g. 1.0.0")
        return value


class GovernorAssetContext(BaseModel):
    """Risk context aligned to a single F37 allocation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    sector: str = Field(min_length=1, max_length=100)
    lot_size: float = Field(gt=0.0)
    average_daily_dollar_volume: float = Field(gt=0.0)
    current_notional: float = Field(default=0.0, ge=0.0)
    factor_loadings: dict[str, float] = Field(default_factory=dict)

    @field_validator("asset_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize symbols before matching or hashing."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("asset_symbol must not be empty")
        return normalized

    @field_validator("sector")
    @classmethod
    def normalize_sector(cls, value: str) -> str:
        """Normalize sector labels for deterministic aggregation."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("sector must not be empty")
        return normalized

    @field_validator("factor_loadings")
    @classmethod
    def validate_factor_loadings(cls, value: dict[str, float]) -> dict[str, float]:
        """Normalize factor names and reject non-finite or duplicated inputs."""
        normalized: dict[str, float] = {}
        for name, loading in value.items():
            key = name.strip().upper()
            if not key:
                raise ValueError("factor names must not be empty")
            if key in normalized:
                raise ValueError(f"duplicate factor name after normalization: {key}")
            if not math.isfinite(loading):
                raise ValueError("factor loadings must be finite")
            normalized[key] = float(loading)
        return dict(sorted(normalized.items()))


class PairwiseCorrelation(BaseModel):
    """Point-in-time correlation value from the F23 dependence layer."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_a: str = Field(min_length=1, max_length=80)
    asset_b: str = Field(min_length=1, max_length=80)
    correlation: float = Field(ge=-1.0, le=1.0)

    @field_validator("asset_a", "asset_b")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Use canonical uppercase symbols for pair identities."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("correlation asset symbols must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_pair(self) -> PairwiseCorrelation:
        """Reject self-correlations, which do not measure cross-asset dependence."""
        if self.asset_a == self.asset_b:
            raise ValueError("pairwise correlation must reference two different assets")
        return self

    @property
    def pair_key(self) -> tuple[str, str]:
        """Return an order-independent correlation pair key."""
        if self.asset_a < self.asset_b:
            return self.asset_a, self.asset_b
        return self.asset_b, self.asset_a


class RiskGovernorRequest(BaseModel):
    """F37 sizing plus aligned risk evidence required before an execution adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sizing_result: F37SizingResult
    assets: tuple[GovernorAssetContext, ...] = Field(min_length=2, max_length=500)
    pairwise_correlations: tuple[PairwiseCorrelation, ...] = Field(default=(), max_length=124750)
    risk_metrics: tuple[RiskMetric, ...] = Field(default=(), max_length=1000)
    market_data_as_of: datetime
    risk_snapshot_as_of: datetime
    as_of: datetime
    data_snapshot_id: str = Field(min_length=1, max_length=200)
    risk_snapshot_id: str = Field(min_length=1, max_length=200)
    model_snapshot_id: str = Field(min_length=1, max_length=200)

    @field_validator("market_data_as_of", "risk_snapshot_as_of", "as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject ambiguous timestamps across market, risk, and decision evidence."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("F38 timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_alignment(self) -> RiskGovernorRequest:
        """Reject universe, currency, fingerprint and duplicate-metric mismatches."""
        source_symbols = tuple(item.asset_symbol.upper() for item in self.sizing_result.allocations)
        context_symbols = tuple(item.asset_symbol for item in self.assets)
        if context_symbols != source_symbols:
            raise ValueError("F38 asset context must exactly match F37 symbol order")
        if len(set(context_symbols)) != len(context_symbols):
            raise ValueError("F38 asset symbols must be unique")
        if self.sizing_result.as_of > self.as_of:
            raise ValueError("F37 sizing timestamp cannot be later than governor as_of")
        if self.market_data_as_of > self.as_of or self.risk_snapshot_as_of > self.as_of:
            raise ValueError("F38 source timestamps cannot be later than governor as_of")
        if math.fsum(item.target_notional for item in self.sizing_result.allocations) > (
            self.sizing_result.equity + 1e-6
        ):
            raise ValueError("F37 notional cannot exceed account equity")
        for allocation, context in zip(
            self.sizing_result.allocations, self.assets, strict=True
        ):
            expected_weight = allocation.target_notional / self.sizing_result.equity
            if not math.isclose(
                allocation.target_weight, expected_weight, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"F37 target weight for {allocation.asset_symbol} "
                    "does not match notional/equity"
                )
            if allocation.target_quantity > 0.0:
                lots = allocation.target_quantity / context.lot_size
                if not math.isclose(lots, round(lots), rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError(
                        f"F37 quantity for {allocation.asset_symbol} "
                        "is not a multiple of F38 lot_size"
                    )

        seen_pairs: set[tuple[str, str]] = set()
        for pair in self.pairwise_correlations:
            if pair.asset_a not in context_symbols or pair.asset_b not in context_symbols:
                raise ValueError("correlation pair references an asset outside F37 sizing universe")
            if pair.pair_key in seen_pairs:
                raise ValueError(f"duplicate pairwise correlation: {pair.pair_key}")
            seen_pairs.add(pair.pair_key)

        reserved_metric_names = {
            "max_position_weight",
            "gross_leverage",
            "concentration_hhi",
            "max_sector_weight",
            "max_abs_factor_exposure",
            "max_abs_correlation",
            "estimated_liquidation_days",
            "turnover_ratio",
        }
        seen_metrics: set[str] = set()
        for metric in self.risk_metrics:
            if metric.name in reserved_metric_names:
                raise ValueError(f"risk metric {metric.name!r} is computed from F38 context")
            if metric.name in seen_metrics:
                raise ValueError(f"duplicate risk evidence metric: {metric.name}")
            seen_metrics.add(metric.name)
        ranges = {
            "max_drawdown": (0.0, 1.0),
            "daily_loss_fraction": (0.0, 1.0),
            "data_quality_score": (0.0, 1.0),
            "model_quality_score": (0.0, 1.0),
        }
        for metric in self.risk_metrics:
            if metric.name in ranges:
                low, high = ranges[metric.name]
                if not low <= metric.value <= high:
                    raise ValueError(f"{metric.name} must be between {low} and {high}")
            if metric.name == "max_cvar_fraction" and metric.value < 0.0:
                raise ValueError("max_cvar_fraction must be non-negative")
        return self


class GovernorApprovedPosition(BaseModel):
    """Position-specific quantities remaining after the F38 risk decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    source_quantity: float = Field(ge=0.0)
    source_notional: float = Field(ge=0.0)
    source_weight: float = Field(ge=0.0, le=1.0)
    approved_quantity: float = Field(ge=0.0)
    approved_notional: float = Field(ge=0.0)
    approved_weight: float = Field(ge=0.0, le=1.0)
    lot_size: float = Field(gt=0.0)
    status: GovernorPositionStatus
    rationale: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_position(self) -> GovernorApprovedPosition:
        """Ensure zero quantity and lots are internally consistent."""
        if self.approved_quantity > self.source_quantity + 1e-8:
            raise ValueError("F38 cannot increase the source F37 quantity")
        if self.approved_notional > self.source_notional + 1e-6:
            raise ValueError("F38 cannot increase the source F37 notional")
        if self.approved_weight > self.source_weight + 1e-8:
            raise ValueError("F38 cannot increase the source F37 weight")
        if self.source_quantity == 0.0 and (
            self.source_notional != 0.0 or self.source_weight != 0.0
        ):
            raise ValueError("zero source quantity must have zero source notional and weight")
        if self.approved_quantity == 0.0:
            if self.approved_notional != 0.0 or self.approved_weight != 0.0:
                raise ValueError("zero approved quantity must have zero notional and weight")
            if self.status not in {"ZERO_SIZE", "BLOCKED"}:
                raise ValueError("zero approved quantity must be ZERO_SIZE or BLOCKED")
        else:
            lots = self.approved_quantity / self.lot_size
            if not math.isclose(lots, round(lots), rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("approved quantity must be a multiple of lot_size")
            if self.status not in {"APPROVED", "REDUCED"}:
                raise ValueError("positive approved quantity must be APPROVED or REDUCED")
        return self


class RiskGovernorResult(BaseModel):
    """Final F38 ALLOW/REDUCE/BLOCK decision; never submits orders."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    action: RiskAction
    reduction_factor: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()
    triggered_constraints: tuple[str, ...] = ()
    metrics: tuple[RiskMetric, ...] = ()
    positions: tuple[GovernorApprovedPosition, ...] = Field(min_length=2, max_length=500)
    equity: float = Field(gt=0.0)
    total_approved_notional: float = Field(ge=0.0)
    total_approved_weight: float = Field(ge=0.0, le=1.0)
    config_id: str = Field(min_length=1, max_length=160)
    config_version: str
    as_of: datetime
    data_snapshot_id: str = Field(min_length=1, max_length=200)
    risk_snapshot_id: str = Field(min_length=1, max_length=200)
    model_snapshot_id: str = Field(min_length=1, max_length=200)
    sizing_fingerprint: str
    governor_fingerprint: str

    @field_validator("as_of")
    @classmethod
    def require_result_timezone(cls, value: datetime) -> datetime:
        """Require a timezone-aware evaluation timestamp."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @field_validator("sizing_fingerprint", "governor_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Require lower-case SHA-256 provenance hashes."""
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("fingerprints must be 64-character lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def validate_result_consistency(self) -> RiskGovernorResult:
        """Make it impossible to represent an approved exposure after a BLOCK."""
        weights = math.fsum(position.approved_weight for position in self.positions)
        notionals = math.fsum(position.approved_notional for position in self.positions)
        if not math.isclose(weights, self.total_approved_weight, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("total_approved_weight must equal sum of approved position weights")
        if not math.isclose(
            notionals, self.total_approved_notional, rel_tol=1e-10, abs_tol=1e-6
        ):
            raise ValueError("total_approved_notional must equal sum of approved notionals")
        if weights > 1.0 + 1e-8 or notionals > self.equity + 1e-6:
            raise ValueError("approved target exposure cannot exceed account equity")
        if self.action == RiskAction.ALLOW and self.reduction_factor != 1.0:
            raise ValueError("ALLOW requires reduction_factor=1.0")
        if self.action == RiskAction.BLOCK:
            if self.reduction_factor != 0.0 or notionals != 0.0:
                raise ValueError("BLOCK requires zero reduction factor and zero approved exposure")
            if any(position.status != "BLOCKED" for position in self.positions):
                raise ValueError("BLOCK must mark every position as BLOCKED")
        if self.action == RiskAction.REDUCE and not 0.0 < self.reduction_factor < 1.0:
            raise ValueError("REDUCE requires reduction_factor strictly between 0 and 1")
        return self


__all__ = [
    "GovernorApprovedPosition",
    "GovernorAssetContext",
    "GovernorPositionStatus",
    "PairwiseCorrelation",
    "RiskGovernorConfig",
    "RiskGovernorRequest",
    "RiskGovernorResult",
]
