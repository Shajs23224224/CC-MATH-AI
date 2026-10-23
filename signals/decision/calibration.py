"""Deterministic temperature calibration from matured out-of-sample outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from itertools import product

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.contracts import SignalType

from ._math import class_probabilities

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DecisionCalibrationSample(BaseModel):
    """One out-of-sample score paired with an outcome observable at fit time."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sample_id: str = Field(min_length=1, max_length=200)
    score: float = Field(ge=-1.0, le=1.0)
    realized_class: SignalType
    horizon: str = Field(min_length=1, max_length=80)
    observed_at: datetime
    label_available_at: datetime
    data_snapshot_id: str = Field(min_length=1, max_length=200)

    @field_validator("observed_at", "label_available_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Refuse ambiguous timestamps in calibration datasets."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibration sample timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_label_timing(self) -> DecisionCalibrationSample:
        """The outcome cannot become available before the prediction was made."""
        if self.label_available_at < self.observed_at:
            raise ValueError("label_available_at must not precede observed_at")
        return self


class ProbabilityCalibrationResult(BaseModel):
    """Fitted parameters and a distinct later temporal validation window."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    calibration_id: str = Field(min_length=1, max_length=160)
    calibration_version: str = Field(min_length=1, max_length=80)
    calibration_data_snapshot_id: str = Field(min_length=1, max_length=200)
    horizon: str = Field(min_length=1, max_length=80)
    fit_as_of: datetime
    calibration_window_start: datetime
    calibration_window_end: datetime
    validation_window_start: datetime
    validation_window_end: datetime
    sample_count: int = Field(ge=30)
    training_sample_count: int = Field(ge=15)
    validation_sample_count: int = Field(ge=9)
    purged_sample_count: int = Field(ge=0)
    class_counts: tuple[tuple[SignalType, int], ...]
    validation_class_counts: tuple[tuple[SignalType, int], ...]
    temperature: float = Field(ge=0.05, le=10.0)
    hold_bias: float = Field(ge=-2.0, le=2.0)
    training_log_loss: float = Field(ge=0.0)
    baseline_log_loss: float = Field(ge=0.0)
    calibrated_log_loss: float = Field(ge=0.0)
    validation_passed: bool
    fit_fingerprint: str

    @field_validator(
        "fit_as_of",
        "calibration_window_start",
        "calibration_window_end",
        "validation_window_start",
        "validation_window_end",
    )
    @classmethod
    def require_window_timezone(cls, value: datetime) -> datetime:
        """Calibration windows are always timezone-aware."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibration timestamps must be timezone-aware")
        return value

    @field_validator("fit_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Require a canonical SHA-256 fingerprint."""
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("fit_fingerprint must be a 64-character lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_window_and_counts(self) -> ProbabilityCalibrationResult:
        """Prevent post-fit timestamps and enforce a purged chronological split."""
        if self.calibration_window_start > self.calibration_window_end:
            raise ValueError("calibration window start must not follow its end")
        if self.calibration_window_end > self.validation_window_start:
            raise ValueError("calibration fit window must end before validation begins")
        if self.validation_window_start > self.validation_window_end:
            raise ValueError("validation window start must not follow its end")
        if self.validation_window_end > self.fit_as_of:
            raise ValueError("validation window must end at or before fit_as_of")
        if (
            self.training_sample_count
            + self.validation_sample_count
            + self.purged_sample_count
            != self.sample_count
        ):
            raise ValueError("training, validation and purged counts must reconcile")
        if self.validation_passed and self.calibrated_log_loss > self.baseline_log_loss + 1e-12:
            raise ValueError("validation cannot pass when calibration worsens log loss")
        return self


def fit_probability_calibration(
    samples: tuple[DecisionCalibrationSample, ...],
    *,
    calibration_id: str,
    calibration_version: str,
    calibration_data_snapshot_id: str,
    fit_as_of: datetime,
    min_samples: int = 30,
) -> ProbabilityCalibrationResult:
    """Fit temperature and HOLD bias on an early temporal slice and validate on a later one.

    Only predictions and labels already observable by fit_as_of are eligible. The
    calibration fit slice is purged so its labels are available by the first validation
    forecast origin. Validation log loss is measured on later samples not used in fitting.
    The labels must follow a separately defined, versioned outcome policy.
    """
    if fit_as_of.tzinfo is None or fit_as_of.utcoffset() is None:
        raise ValueError("fit_as_of must be timezone-aware")
    if min_samples < 30:
        raise ValueError("min_samples must be at least 30")
    if len(samples) < min_samples:
        raise ValueError(f"at least {min_samples} calibration samples are required")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("calibration sample IDs must be unique")

    horizons = {sample.horizon for sample in samples}
    if len(horizons) != 1:
        raise ValueError("all calibration samples must use the same horizon")
    for sample in samples:
        if sample.observed_at > fit_as_of:
            raise ValueError("calibration forecast observed_at is later than fit_as_of")
        if sample.label_available_at > fit_as_of:
            raise ValueError("calibration label is not yet observable at fit_as_of")

    ordered = tuple(sorted(samples, key=lambda sample: (sample.observed_at, sample.sample_id)))
    split_index = int(len(ordered) * 0.7)
    training_candidates = ordered[:split_index]
    validation_samples = ordered[split_index:]
    if not validation_samples:
        raise ValueError("calibration requires a non-empty later validation slice")
    validation_start = min(sample.observed_at for sample in validation_samples)
    training_samples = tuple(
        sample for sample in training_candidates if sample.label_available_at <= validation_start
    )
    purged_count = len(training_candidates) - len(training_samples)
    if len(training_samples) < max(15, (min_samples * 2) // 3):
        raise ValueError("too few training samples remain after temporal purging")
    if len(validation_samples) < max(9, min_samples // 5):
        raise ValueError("too few later validation samples")
    _require_class_coverage(training_samples, "training")
    _require_class_coverage(validation_samples, "validation")

    baseline_temperature = 0.5
    baseline_hold_bias = 0.5
    baseline_validation_loss = _mean_log_loss(
        validation_samples, baseline_temperature, baseline_hold_bias
    )

    log_min = math.log(0.05)
    log_max = math.log(10.0)
    temperatures = {
        math.exp(log_min + (log_max - log_min) * index / 32)
        for index in range(33)
    }
    temperatures.update({baseline_temperature, 1.0, 10.0})
    hold_biases = {-2.0 + index * 0.1 for index in range(41)}
    hold_biases.add(baseline_hold_bias)

    best: tuple[float, float, float] | None = None
    for temperature, hold_bias in product(sorted(temperatures), sorted(hold_biases)):
        training_loss = _mean_log_loss(training_samples, temperature, hold_bias)
        candidate = (training_loss, temperature, hold_bias)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    training_loss, temperature, hold_bias = best
    validation_loss = _mean_log_loss(validation_samples, temperature, hold_bias)
    validation_end = max(sample.label_available_at for sample in validation_samples)

    canonical_samples = sorted(
        (sample.model_dump(mode="json") for sample in samples),
        key=lambda item: item["sample_id"],
    )
    canonical = json.dumps(
        {
            "samples": canonical_samples,
            "fit_as_of": fit_as_of.isoformat(),
            "calibration_id": calibration_id,
            "calibration_version": calibration_version,
            "calibration_data_snapshot_id": calibration_data_snapshot_id,
            "min_samples": min_samples,
            "temporal_training_fraction": 0.7,
            "grid": {"temperature": [0.05, 10.0, 33], "hold_bias": [-2.0, 2.0, 0.1]},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return ProbabilityCalibrationResult(
        calibration_id=calibration_id,
        calibration_version=calibration_version,
        calibration_data_snapshot_id=calibration_data_snapshot_id,
        horizon=next(iter(horizons)),
        fit_as_of=fit_as_of,
        calibration_window_start=min(sample.observed_at for sample in training_samples),
        calibration_window_end=max(sample.label_available_at for sample in training_samples),
        validation_window_start=validation_start,
        validation_window_end=validation_end,
        sample_count=len(samples),
        training_sample_count=len(training_samples),
        validation_sample_count=len(validation_samples),
        purged_sample_count=purged_count,
        class_counts=_class_counts(training_samples),
        validation_class_counts=_class_counts(validation_samples),
        temperature=temperature,
        hold_bias=hold_bias,
        training_log_loss=training_loss,
        baseline_log_loss=baseline_validation_loss,
        calibrated_log_loss=validation_loss,
        validation_passed=validation_loss <= baseline_validation_loss + 1e-12,
        fit_fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _require_class_coverage(
    samples: tuple[DecisionCalibrationSample, ...],
    sample_slice: str,
) -> None:
    """Require evidence for all three probability classes in each temporal slice."""
    counts = {signal: 0 for signal in SignalType}
    for sample in samples:
        counts[sample.realized_class] += 1
    if any(counts[signal] < 2 for signal in SignalType):
        raise ValueError(f"{sample_slice} slice requires at least two samples for each class")


def _class_counts(
    samples: tuple[DecisionCalibrationSample, ...],
) -> tuple[tuple[SignalType, int], ...]:
    """Return class frequencies in stable enum order."""
    counts = {signal: 0 for signal in SignalType}
    for sample in samples:
        counts[sample.realized_class] += 1
    return tuple((signal, counts[signal]) for signal in SignalType)


def _mean_log_loss(
    samples: tuple[DecisionCalibrationSample, ...],
    temperature: float,
    hold_bias: float,
) -> float:
    """Compute the deterministic multiclass negative log likelihood."""
    losses: list[float] = []
    class_index = {
        SignalType.BUY: 0,
        SignalType.HOLD: 1,
        SignalType.SELL: 2,
    }
    for sample in samples:
        probabilities = class_probabilities(sample.score, temperature, hold_bias)
        losses.append(-math.log(max(probabilities[class_index[sample.realized_class]], 1e-15)))
    return math.fsum(losses) / len(losses)


__all__ = [
    "DecisionCalibrationSample",
    "ProbabilityCalibrationResult",
    "fit_probability_calibration",
]
