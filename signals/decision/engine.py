"""Deterministic meta-decision logic; this module never places orders."""

from __future__ import annotations

import hashlib
import json
import math

from core.contracts import DecisionRecord, SignalType
from signals.ensemble import EnsembleResult

from ._math import class_probabilities

from .contracts import (
    DecisionConfig,
    DecisionEvidence,
    DecisionOutput,
    DecisionProbabilities,
    DecisionRequest,
)


def make_decision(
    request: DecisionRequest,
    config: DecisionConfig | None = None,
) -> DecisionOutput:
    """Build a replayable decision record from F34 evidence and a matched forecast.

    Probabilities are produced by a versioned temperature-scaled softmax. They are
    explicitly labelled uncalibrated unless the caller supplies validated out-of-sample
    calibration metadata. With the default policy, uncalibrated probabilities abstain.
    """
    policy = config or DecisionConfig()
    ensemble = request.ensemble
    probabilities = _probabilities(ensemble.score, policy)
    forecast_uncertainty = min(
        1.0, request.forecast.uncertainty / policy.forecast_uncertainty_scale
    )
    combined_uncertainty = 1.0 - (
        (1.0 - ensemble.uncertainty) * (1.0 - forecast_uncertainty)
    )
    combined_uncertainty = min(1.0, max(0.0, combined_uncertainty))

    abstention_reason = _abstention_reason(
        request=request,
        policy=policy,
        uncertainty=combined_uncertainty,
    )
    signal = SignalType.HOLD
    if abstention_reason is None:
        if ensemble.score >= policy.buy_threshold:
            if policy.require_forecast_alignment and request.forecast.expected_return <= 0.0:
                abstention_reason = "buy_signal_conflicts_with_non_positive_forecast_return"
            elif probabilities.buy < policy.minimum_action_probability:
                abstention_reason = "buy_probability_below_action_threshold"
            else:
                signal = SignalType.BUY
        elif ensemble.score <= policy.sell_threshold:
            if policy.require_forecast_alignment and request.forecast.expected_return >= 0.0:
                abstention_reason = "sell_signal_conflicts_with_non_negative_forecast_return"
            elif probabilities.sell < policy.minimum_action_probability:
                abstention_reason = "sell_probability_below_action_threshold"
            else:
                signal = SignalType.SELL

    abstained = abstention_reason is not None
    if abstained:
        signal = SignalType.HOLD

    selected_probability = probabilities.for_signal(signal)
    confidence = (
        0.0
        if abstained
        else min(
            1.0,
            max(0.0, selected_probability * ensemble.agreement * (1.0 - combined_uncertainty)),
        )
    )
    risk_adjusted_score = request.forecast.expected_return / max(
        request.forecast.expected_risk, policy.minimum_risk_floor
    )
    if not math.isfinite(risk_adjusted_score):
        risk_adjusted_score = 0.0
        if not abstained:
            abstention_reason = "non_finite_risk_adjusted_score"
            abstained = True
            signal = SignalType.HOLD
            selected_probability = probabilities.hold
            confidence = 0.0

    decision_fingerprint = _fingerprint(request, policy)
    record = DecisionRecord(
        decision_id=f"dec-{decision_fingerprint[:24]}",
        asset_symbol=ensemble.asset_symbol,
        timestamp=ensemble.as_of,
        signal=signal,
        signal_probability=selected_probability,
        model_confidence=confidence,
        expected_return=request.forecast.expected_return,
        expected_risk=request.forecast.expected_risk,
        risk_adjusted_score=risk_adjusted_score,
        recommended_horizon=ensemble.horizon,
        # Portfolio construction and position sizing are deliberately downstream.
        suggested_position_size=0.0,
        regime=request.regime,
        data_snapshot_id=ensemble.data_snapshot_id,
        model_versions=tuple(
            sorted(
                set(request.model_versions)
                | {
                    f"forecast:{request.forecast_source_id}@{request.forecast_source_version}",
                    f"ensemble:{ensemble.config_id}@{ensemble.config_version}",
                }
            )
        ),
        feature_versions=tuple(sorted(set(request.feature_versions))),
    )
    evidence = tuple(
        DecisionEvidence(
            source_id=item.source_id,
            source_version=item.source_version,
            correlation_group=item.correlation_group,
            direction=1 if item.normalized_score > 0 else -1 if item.normalized_score < 0 else 0,
            normalized_score=item.normalized_score,
            contribution=item.contribution,
            rationale=item.rationale,
            observed_at=item.observed_at,
        )
        for item in sorted(ensemble.contributions, key=lambda item: (item.source_id, item.source_version))
    )

    return DecisionOutput(
        record=record,
        probabilities=probabilities,
        status="ABSTAINED" if abstained else "DECIDED",
        abstained=abstained,
        abstention_reason=abstention_reason,
        agreement=ensemble.agreement,
        uncertainty=combined_uncertainty,
        evidence=evidence,
        constraints=tuple(sorted(request.constraints, key=lambda item: item.constraint_id)),
        decision_config_id=policy.config_id,
        decision_config_version=policy.version,
        probability_calibration_status=policy.probability_calibration_status,
        calibration_id=policy.calibration_id,
        calibration_fit_fingerprint=policy.calibration_fit_fingerprint,
        calibration_version=policy.calibration_version,
        calibration_data_snapshot_id=policy.calibration_data_snapshot_id,
        calibration_window_start=policy.calibration_window_start,
        calibration_window_end=policy.calibration_window_end,
        calibration_validation_data_snapshot_id=policy.calibration_validation_data_snapshot_id,
        calibration_validation_window_start=policy.calibration_validation_window_start,
        calibration_validation_window_end=policy.calibration_validation_window_end,
        calibration_validation_log_loss=policy.calibration_validation_log_loss,
        calibration_validation_baseline_log_loss=policy.calibration_validation_baseline_log_loss,
        ensemble_fingerprint=ensemble.input_fingerprint,
        decision_fingerprint=decision_fingerprint,
    )


def _probabilities(score: float, config: DecisionConfig) -> DecisionProbabilities:
    """Map the bounded ensemble score to a stable three-class distribution."""
    buy, hold, sell = class_probabilities(
        score,
        config.probability_temperature,
        config.hold_bias,
    )
    return DecisionProbabilities(buy=buy, hold=hold, sell=sell)


def _abstention_reason(
    request: DecisionRequest,
    policy: DecisionConfig,
    uncertainty: float,
) -> str | None:
    """Apply deterministic reject gates in stable priority order."""
    ensemble = request.ensemble
    if ensemble.abstained:
        return f"ensemble_abstained:{ensemble.abstention_reason or 'unspecified'}"
    if (
        policy.require_calibrated_probabilities
        and policy.probability_calibration_status != "validated_oos"
    ):
        return "probabilities_not_validated_out_of_sample"
    if policy.probability_calibration_status == "validated_oos":
        assert policy.calibration_window_end is not None
        assert policy.calibration_validation_window_end is not None
        if policy.calibration_window_end > ensemble.as_of:
            return "calibration_window_after_decision_as_of"
        if policy.calibration_validation_window_end > ensemble.as_of:
            return "calibration_validation_after_decision_as_of"
    failed_constraints = sorted(
        item.constraint_id
        for item in request.constraints
        if item.blocking and not item.satisfied
    )
    if failed_constraints:
        return "blocking_constraints_failed:" + ",".join(failed_constraints)
    if ensemble.agreement < policy.minimum_agreement:
        return "agreement_below_minimum"
    if uncertainty > policy.maximum_uncertainty:
        return "uncertainty_above_maximum"
    return None


def _fingerprint(request: DecisionRequest, config: DecisionConfig) -> str:
    """Hash canonical request and versioned policy metadata for exact replay."""
    request_payload = request.model_dump(mode="json")
    request_payload["model_versions"] = sorted(request_payload["model_versions"])
    request_payload["feature_versions"] = sorted(request_payload["feature_versions"])
    request_payload["constraints"] = sorted(
        request_payload["constraints"], key=lambda item: item["constraint_id"]
    )
    ensemble_payload = request_payload["ensemble"]
    ensemble_payload["contributions"] = sorted(
        ensemble_payload["contributions"],
        key=lambda item: (item["source_id"], item["source_version"]),
    )
    ensemble_payload["excluded_evidence"] = sorted(
        ensemble_payload["excluded_evidence"],
        key=lambda item: (item["source_id"], item["source_version"], item["reason"]),
    )
    canonical = json.dumps(
        {"request": request_payload, "config": config.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["make_decision"]
