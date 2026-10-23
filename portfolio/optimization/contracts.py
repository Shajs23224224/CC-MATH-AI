"""Typed F36 portfolio optimization contracts and acceptance diagnostics."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.contracts import DependenceMatrix, SignalType, TargetAllocation
from signals.decision import DecisionOutput

_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

OptimizationMethod = Literal[
    "equal_weight",
    "mean_variance",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
    "inverse_volatility",
    "hrp",
    "black_litterman",
    "maximum_diversification",
]


class SectorLimit(BaseModel):
    """Long-only upper bound for the sum of weights in one sector."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sector: str = Field(min_length=1, max_length=100)
    maximum_weight: float = Field(gt=0.0, le=1.0)

    @field_validator("sector")
    @classmethod
    def normalize_sector(cls, value: str) -> str:
        """Normalize sector labels for deterministic matching."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("sector must not be blank")
        return normalized


class PortfolioOptimizationConfig(BaseModel):
    """Versioned optimizer policy; returns and covariance must share one horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    config_id: str = Field(default="portfolio-optimizer", min_length=1, max_length=160)
    version: str = "1.0.0"
    method: OptimizationMethod = "mean_variance"
    risk_aversion: float = Field(default=3.0, gt=0.0, le=1_000_000.0)
    risk_free_rate: float = Field(default=0.0, ge=-1.0, le=10.0)
    black_litterman_tau: float = Field(default=0.05, gt=0.0, le=1.0)
    transaction_cost_bps: float = Field(default=0.0, ge=0.0, le=10_000.0)
    turnover_penalty: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    max_turnover: float | None = Field(default=None, ge=0.0, le=1.0)
    max_concentration_hhi: float | None = Field(default=None, gt=0.0, le=1.0)
    sector_limits: tuple[SectorLimit, ...] = Field(default=(), max_length=500)
    covariance_ridge: float = Field(default=1e-10, ge=0.0, le=0.01)
    max_iterations: int = Field(default=1000, ge=50, le=20_000)
    solver_tolerance: float = Field(default=1e-10, gt=0.0, le=0.01)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Require semantic versions for replayable optimizer policy references."""
        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("version must use semantic versioning, e.g. 1.0.0")
        return value

    @model_validator(mode="after")
    def validate_sector_limits(self) -> PortfolioOptimizationConfig:
        """Reject ambiguous duplicate sector caps."""
        sectors = [item.sector for item in self.sector_limits]
        if len(sectors) != len(set(sectors)):
            raise ValueError("sector_limits must not contain duplicate normalized sectors")
        return self


class PortfolioAssetInput(BaseModel):
    """One point-in-time F35 decision plus portfolio constraints for that asset."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    decision: DecisionOutput
    current_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    sector: str | None = Field(default=None, max_length=100)
    eligible: bool = True

    @field_validator("sector")
    @classmethod
    def normalize_sector(cls, value: str | None) -> str | None:
        """Normalize optional sector labels before cap evaluation."""
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("sector must be null or non-blank")
        return normalized

    @model_validator(mode="after")
    def validate_asset_limits(self) -> PortfolioAssetInput:
        """Reject contradictory per-asset allocation bounds."""
        if self.minimum_weight > self.maximum_weight:
            raise ValueError("minimum_weight must not exceed maximum_weight")
        return self

    @property
    def asset_symbol(self) -> str:
        """Use the canonical asset identity from the upstream F35 record."""
        return self.decision.record.asset_symbol


class PortfolioOptimizationRequest(BaseModel):
    """Time-scoped F35 decision set and matching F23 covariance matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    assets: tuple[PortfolioAssetInput, ...] = Field(min_length=2, max_length=500)
    covariance: DependenceMatrix
    as_of: datetime
    horizon: str = Field(min_length=1, max_length=80)
    covariance_as_of: datetime
    covariance_horizon: str = Field(min_length=1, max_length=80)
    data_snapshot_id: str = Field(min_length=1, max_length=200)
    covariance_data_snapshot_id: str = Field(min_length=1, max_length=200)
    view_matrix: tuple[tuple[float, ...], ...] = Field(default=(), max_length=500)
    view_returns: tuple[float, ...] = Field(default=(), max_length=500)
    view_confidences: tuple[float, ...] = Field(default=(), max_length=500)
    equilibrium_weights: tuple[float, ...] = ()

    @field_validator("as_of", "covariance_as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject ambiguous information cutoffs."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("portfolio and covariance timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> PortfolioOptimizationRequest:
        """Enforce symbol ordering, covariance dimensions, horizon and causal scope."""
        if self.covariance.kind != "covariance":
            raise ValueError("F36 requires a covariance matrix, not a correlation matrix")
        symbols = tuple(asset.asset_symbol for asset in self.assets)
        if len(set(symbols)) != len(symbols):
            raise ValueError("decision asset symbols must be unique")
        if symbols != self.covariance.assets:
            raise ValueError(
                "assets must be sorted and exactly match covariance asset order"
            )
        if self.horizon != self.covariance_horizon:
            raise ValueError("forecast and covariance horizons must match")
        if self.covariance_as_of > self.as_of:
            raise ValueError("covariance_as_of must not be later than portfolio as_of")
        if any(asset.decision.record.timestamp > self.as_of for asset in self.assets):
            raise ValueError("F35 decision timestamps must not be later than portfolio as_of")
        if any(asset.decision.record.recommended_horizon != self.horizon for asset in self.assets):
            raise ValueError("all F35 decisions must use the requested portfolio horizon")
        if any(
            not math.isfinite(asset.decision.record.expected_return)
            or not math.isfinite(asset.decision.record.expected_risk)
            for asset in self.assets
        ):
            raise ValueError("F35 expected return and risk must be finite")
        if sum(asset.current_weight for asset in self.assets) > 1.0 + 1e-8:
            raise ValueError("current long-only asset weights must not exceed one")

        count = len(self.assets)
        if self.equilibrium_weights:
            if len(self.equilibrium_weights) != count:
                raise ValueError("equilibrium_weights must match asset count")
            if any(not math.isfinite(value) or value < 0.0 for value in self.equilibrium_weights):
                raise ValueError("equilibrium_weights must be finite and non-negative")
            if not math.isclose(
                math.fsum(self.equilibrium_weights), 1.0, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError("equilibrium_weights must sum to one")

        view_count = len(self.view_returns)
        if len(self.view_matrix) != view_count or len(self.view_confidences) != view_count:
            raise ValueError("view_matrix, view_returns and view_confidences must align")
        if any(len(row) != count for row in self.view_matrix):
            raise ValueError("each Black-Litterman view row must match asset count")
        if any(not math.isfinite(value) for row in self.view_matrix for value in row):
            raise ValueError("view_matrix entries must be finite")
        if any(not math.isfinite(value) for value in self.view_returns):
            raise ValueError("view_returns must be finite")
        if any(not any(abs(value) > 1e-12 for value in row) for row in self.view_matrix):
            raise ValueError("each Black-Litterman view must reference at least one asset")
        if any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.view_confidences
        ):
            raise ValueError("view_confidences must be finite and in (0, 1]")
        covariance_values = tuple(value for row in self.covariance.matrix for value in row)
        if any(not math.isfinite(value) for value in covariance_values):
            raise ValueError("covariance matrix must be finite")
        if any(self.covariance.matrix[index][index] <= 0.0 for index in range(count)):
            raise ValueError("each eligible covariance diagonal must be strictly positive")
        return self


class AssetExclusion(BaseModel):
    """Reason an asset received a zero target weight."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=500)


class ConstraintDiagnostic(BaseModel):
    """Measured portfolio constraint and explicit pass/fail result."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    constraint_id: str = Field(min_length=1, max_length=180)
    satisfied: bool
    measured: float
    limit: float | None = None
    detail: str = Field(min_length=1, max_length=500)


class PortfolioOptimizationResult(BaseModel):
    """Auditable target allocations, solver outcome and portfolio diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    status: Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "FAILED"]
    method: OptimizationMethod
    config_id: str = Field(min_length=1, max_length=160)
    config_version: str
    asset_symbols: tuple[str, ...] = Field(min_length=2, max_length=500)
    allocations: tuple[TargetAllocation, ...] = Field(max_length=500)
    exclusions: tuple[AssetExclusion, ...] = Field(default=(), max_length=500)
    constraints: tuple[ConstraintDiagnostic, ...] = Field(default=(), max_length=2000)
    objective_value: float | None = None
    expected_return: float | None = None
    volatility: float | None = Field(default=None, ge=0.0)
    sharpe_ratio: float | None = None
    turnover: float | None = Field(default=None, ge=0.0)
    estimated_transaction_cost_fraction: float | None = Field(default=None, ge=0.0)
    concentration_hhi: float | None = Field(default=None, ge=0.0, le=1.0)
    covariance_regularization: float = Field(ge=0.0)
    solver_success: bool
    solver_message: str = Field(min_length=1, max_length=2000)
    iterations: int = Field(ge=0)
    as_of: datetime
    horizon: str = Field(min_length=1, max_length=80)
    data_snapshot_id: str = Field(min_length=1, max_length=200)
    covariance_data_snapshot_id: str = Field(min_length=1, max_length=200)
    decision_fingerprints: tuple[str, ...] = Field(min_length=2, max_length=500)
    optimization_fingerprint: str

    @field_validator("as_of")
    @classmethod
    def require_result_timezone(cls, value: datetime) -> datetime:
        """Keep result timestamps explicit."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @field_validator("optimization_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Require a canonical lowercase SHA-256 fingerprint."""
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("optimization_fingerprint must be a 64-character lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_result_consistency(self) -> PortfolioOptimizationResult:
        """Never label a failed, infeasible or constraint-breaching result as valid."""
        if len(self.asset_symbols) != len(set(self.asset_symbols)):
            raise ValueError("asset_symbols must be unique")
        allocation_symbols = tuple(item.asset_symbol for item in self.allocations)
        if allocation_symbols != self.asset_symbols:
            raise ValueError("allocations must include all assets in deterministic request order")
        if len(set(self.decision_fingerprints)) != len(self.decision_fingerprints):
            raise ValueError("decision_fingerprints must be unique")
        for fingerprint in self.decision_fingerprints:
            if not _SHA256_PATTERN.fullmatch(fingerprint):
                raise ValueError("decision fingerprints must be lowercase SHA-256")
        if self.status in {"OPTIMAL", "FEASIBLE"}:
            total = math.fsum(item.target_weight for item in self.allocations)
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("feasible allocations must sum to one")
            if self.expected_return is None or self.volatility is None or self.turnover is None:
                raise ValueError("feasible outputs require portfolio diagnostics")
            if any(not item.satisfied for item in self.constraints):
                raise ValueError("feasible outputs must satisfy every reported constraint")
        return self


__all__ = [
    "AssetExclusion",
    "ConstraintDiagnostic",
    "OptimizationMethod",
    "PortfolioAssetInput",
    "PortfolioOptimizationConfig",
    "PortfolioOptimizationRequest",
    "PortfolioOptimizationResult",
    "SectorLimit",
]
