"""Typed, versioned contracts for the F35 meta-model decision engine."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.contracts import DecisionRecord, Forecast, SignalType
from signals.ensemble import EnsembleResult

_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DecisionProbabilities(BaseModel):
    """Normalized class probabilities for BUY, HOLD and SELL."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    buy: float = Field(ge=0.0, le=1.0)
    hold: float = Field(ge=0.0, le=1.0)
    sell: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> DecisionProbabilities:
        """Require a complete probability distribution."""
        total = math.fsum((self.buy, self.hold, self.sell))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("BUY/HOLD/SELL probabilities must sum to one")
        return self

    def for_signal(self, signal: SignalType) -> float:
        """Return the probability associated with a selected signal."""
        return {
            SignalType.BUY: self.buy,
            SignalType.HOLD: self.hold,
            SignalType.SELL: self.sell,
        }[signal]


class DecisionConstraint(BaseModel):
    """A pre-decision eligibility constraint; this is not the Risk Governor."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    constraint_id: str = Field(min_length=1, max_length=160)
    satisfied: bool
    blocking: bool = True
    reason: str = Field(min_length=1, max_length=500)


class DecisionEvidence(BaseModel):
    """One F34 contribution copied into the decision audit trail."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source_id: str = Field(min_length=1, max_length=160)
    source_version: str = Field(min_length=1, max_length=80)
    correlation_group: str = Field(min_length=1, max_length=160)
    direction: int = Field(ge=-1, le=1)
    normalized_score: float = Field(ge=-1.0, le=1.0)
    contribution: float = Field(ge=-1.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=1000)
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject ambiguous local timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value


class DecisionConfig(BaseModel):
    """Versioned policy and probability calibration metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    config_id: str = Field(default="meta-decision", min_length=1, max_length=160)
    version: str = "1.0.0"
    buy_threshold: float = Field(default=0.25, gt=0.0, le=1.0)
    sell_threshold: float = Field(default=-0.25, ge=-1.0, lt=0.0)
    minimum_agreement: float = Field(default=0.55, ge=0.0, le=1.0)
    maximum_uncertainty: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_action_probability: float = Field(default=0.55, ge=0.0, le=1.0)
    probability_temperature: float = Field(default=0.5, gt=0.0, le=100.0)
    hold_bias: float = Field(default=0.5, ge=-20.0, le=20.0)
    minimum_risk_floor: float = Field(default=1e-6, gt=0.0, le=1.0)
    forecast_uncertainty_scale: float = Field(default=1.0, gt=0.0, le=1_000_000.0)
    probability_calibration_status: Literal["uncalibrated", "validated_oos"] = "uncalibrated"
    calibration_id: str | None = Field(default=None, min_length=1, max_length=160)
    calibration_fit_fingerprint: str | None = None
    calibration_version: str | None = Field(default=None, min_length=1, max_length=80)
    calibration_data_snapshot_id: str | None = Field(default=None, min_length=1, max_length=200)
    calibration_window_start: datetime | None = None
    calibration_window_end: datetime | None = None
    calibration_validation_data_snapshot_id: str | None = Field(
        default=None, min_length=1, max_length=200
    )
    calibration_validation_window_start: datetime | None = None
    calibration_validation_window_end: datetime | None = None
    calibration_validation_log_loss: float | None = Field(default=None, ge=0.0)
    calibration_validation_baseline_log_loss: float | None = Field(default=None, ge=0.0)
    require_calibrated_probabilities: bool = True
    require_forecast_alignment: bool = True

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Use semantic versions for replayable policy references."""
        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("version must use semantic versioning, e.g. 1.0.0")
        return value

    @field_validator("calibration_fit_fingerprint")
    @classmethod
    def validate_calibration_fingerprint(cls, value: str | None) -> str | None:
        """Require a SHA-256 reference to the fitted calibrator artifact."""
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("calibration_fit_fingerprint must be a lowercase SHA-256")
        return value

    @field_validator("calibration_version")
    @classmethod
    def validate_calibration_version(cls, value: str | None) -> str | None:
        """Validate an optional calibration artifact version."""
        if value is not None and not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("calibration_version must use semantic versioning, e.g. 1.0.0")
        return value

    @field_validator(
        "calibration_window_start",
        "calibration_window_end",
        "calibration_validation_window_start",
        "calibration_validation_window_end",
    )
    @classmethod
    def require_calibration_timezone(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware calibration boundaries when supplied."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("calibration window timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_policy(self) -> DecisionConfig:
        """Prevent ambiguous thresholds and undocumented calibrated probabilities."""
        if self.sell_threshold >= self.buy_threshold:
            raise ValueError("sell_threshold must be lower than buy_threshold")
        if self.probability_calibration_status == "validated_oos":
            required = (
                self.calibration_id,
                self.calibration_fit_fingerprint,
                self.calibration_version,
                self.calibration_data_snapshot_id,
                self.calibration_window_start,
                self.calibration_window_end,
                self.calibration_validation_data_snapshot_id,
                self.calibration_validation_window_start,
                self.calibration_validation_window_end,
                self.calibration_validation_log_loss,
                self.calibration_validation_baseline_log_loss,
            )
            if any(value is None for value in required):
                raise ValueError("validated_oos calibration requires fit and validation provenance")
            assert self.calibration_window_start is not None
            assert self.calibration_window_end is not None
            assert self.calibration_validation_window_start is not None
            assert self.calibration_validation_window_end is not None
            assert self.calibration_validation_log_loss is not None
            assert self.calibration_validation_baseline_log_loss is not None
            if self.calibration_window_start > self.calibration_window_end:
                raise ValueError("calibration window start must not follow its end")
            if self.calibration_window_end > self.calibration_validation_window_start:
                raise ValueError("calibration fit window must end before validation begins")
            if self.calibration_validation_window_start > self.calibration_validation_window_end:
                raise ValueError("validation window start must not follow its end")
            if (
                self.calibration_validation_log_loss
                > self.calibration_validation_baseline_log_loss + 1e-12
            ):
                raise ValueError("validated_oos calibration must not worsen validation log loss")
        elif any(
            value is not None
            for value in (
                self.calibration_id,
                self.calibration_fit_fingerprint,
                self.calibration_version,
                self.calibration_data_snapshot_id,
                self.calibration_window_start,
                self.calibration_window_end,
                self.calibration_validation_data_snapshot_id,
                self.calibration_validation_window_start,
                self.calibration_validation_window_end,
                self.calibration_validation_log_loss,
                self.calibration_validation_baseline_log_loss,
            )
        ):
            raise ValueError("uncalibrated policy must not claim calibration artifact metadata")
        return self


class DecisionRequest(BaseModel):
    """Ensemble, horizon-matched forecast and explicit pre-decision checks."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    ensemble: EnsembleResult
    forecast: Forecast
    forecast_source_id: str = Field(min_length=1, max_length=160)
    forecast_source_version: str = Field(min_length=1, max_length=80)
    forecast_observed_at: datetime
    regime: str = Field(default="UNKNOWN", min_length=1, max_length=100)
    model_versions: tuple[str, ...] = Field(default=(), max_length=500)
    feature_versions: tuple[str, ...] = Field(default=(), max_length=500)
    constraints: tuple[DecisionConstraint, ...] = Field(default=(), max_length=500)

    @field_validator("forecast_observed_at")
    @classmethod
    def require_forecast_timezone(cls, value: datetime) -> datetime:
        """Refuse forecast timestamps without timezone information."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("forecast_observed_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_forecast_scope(self) -> DecisionRequest:
        """Prevent use of a mismatched horizon, future forecast or invalid numerics."""
        if self.forecast.horizon != self.ensemble.horizon:
            raise ValueError("forecast horizon must match the ensemble horizon")
        if self.forecast_observed_at > self.ensemble.as_of:
            raise ValueError("forecast_observed_at must not be later than ensemble as_of")
        if not math.isfinite(self.forecast.expected_return):
            raise ValueError("forecast expected_return must be finite")
        if not math.isfinite(self.forecast.expected_risk):
            raise ValueError("forecast expected_risk must be finite")
        if not math.isfinite(self.forecast.uncertainty):
            raise ValueError("forecast uncertainty must be finite")
        constraint_ids = [item.constraint_id for item in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint IDs must be unique")
        return self


class DecisionOutput(BaseModel):
    """Auditable decision record plus class probabilities and F34 provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    record: DecisionRecord
    probabilities: DecisionProbabilities
    status: Literal["DECIDED", "ABSTAINED"]
    abstained: bool
    abstention_reason: str | None = Field(default=None, max_length=500)
    agreement: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence: tuple[DecisionEvidence, ...] = Field(default=(), max_length=1000)
    constraints: tuple[DecisionConstraint, ...] = Field(default=(), max_length=500)
    decision_config_id: str = Field(min_length=1, max_length=160)
    decision_config_version: str = Field(min_length=1, max_length=80)
    probability_calibration_status: Literal["uncalibrated", "validated_oos"]
    calibration_id: str | None = Field(default=None, max_length=160)
    calibration_fit_fingerprint: str | None = None
    calibration_version: str | None = None
    calibration_data_snapshot_id: str | None = None
    calibration_window_start: datetime | None = None
    calibration_window_end: datetime | None = None
    calibration_validation_data_snapshot_id: str | None = None
    calibration_validation_window_start: datetime | None = None
    calibration_validation_window_end: datetime | None = None
    calibration_validation_log_loss: float | None = None
    calibration_validation_baseline_log_loss: float | None = None
    ensemble_fingerprint: str
    decision_fingerprint: str

    @field_validator(
        "calibration_window_start",
        "calibration_window_end",
        "calibration_validation_window_start",
        "calibration_validation_window_end",
    )
    @classmethod
    def validate_calibration_timestamps(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware audit timestamps when calibration metadata is present."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("calibration timestamps must be timezone-aware")
        return value

    @field_validator("ensemble_fingerprint", "decision_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Require canonical SHA-256 fingerprints."""
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("fingerprint must be a 64-character lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_output_consistency(self) -> DecisionOutput:
        """Keep signal, abstention state and probabilities consistent."""
        calibration_fields = (
            self.calibration_id,
            self.calibration_fit_fingerprint,
            self.calibration_version,
            self.calibration_data_snapshot_id,
            self.calibration_window_start,
            self.calibration_window_end,
            self.calibration_validation_data_snapshot_id,
            self.calibration_validation_window_start,
            self.calibration_validation_window_end,
            self.calibration_validation_log_loss,
            self.calibration_validation_baseline_log_loss,
        )
        if self.probability_calibration_status == "validated_oos":
            if any(value is None for value in calibration_fields):
                raise ValueError("validated_oos output requires complete calibration provenance")
            assert self.calibration_window_end is not None
            assert self.calibration_validation_window_start is not None
            assert self.calibration_validation_window_end is not None
            assert self.calibration_validation_log_loss is not None
            assert self.calibration_validation_baseline_log_loss is not None
            if self.calibration_window_end > self.calibration_validation_window_start:
                raise ValueError("calibration fit window must end before validation begins")
            if (
                self.calibration_validation_log_loss
                > self.calibration_validation_baseline_log_loss + 1e-12
            ):
                raise ValueError("validated_oos output cannot report worse validation log loss")
        elif any(value is not None for value in calibration_fields):
            raise ValueError("uncalibrated output must not carry calibration artifact metadata")
        if self.abstained != (self.status == "ABSTAINED"):
            raise ValueError("status must agree with abstained")
        if self.abstained and not self.abstention_reason:
            raise ValueError("abstained decisions require a reason")
        if not self.abstained and self.abstention_reason is not None:
            raise ValueError("decided outputs must not carry an abstention reason")
        if self.abstained and self.record.signal != SignalType.HOLD:
            raise ValueError("abstained records must use HOLD and remain non-actionable")
        selected_probability = self.probabilities.for_signal(self.record.signal)
        if not math.isclose(
            self.record.signal_probability, selected_probability, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("record signal_probability must match the selected class probability")
        contribution_ids = [item.source_id for item in self.evidence]
        if len(contribution_ids) != len(set(contribution_ids)):
            raise ValueError("decision evidence source IDs must be unique")
        return self


__all__ = [
    "DecisionConfig",
    "DecisionConstraint",
    "DecisionEvidence",
    "DecisionOutput",
    "DecisionProbabilities",
    "DecisionRequest",
]
