"""Behavioral tests for F35 deterministic decisioning and abstention."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from core.contracts import Forecast, SignalType
from signals.decision import (
    DecisionConfig,
    DecisionConstraint,
    DecisionRequest,
    make_decision,
)
from signals.ensemble import EnsembleEvidence, EnsembleRequest, combine_evidence

AS_OF = datetime(2026, 10, 23, 15, 0, tzinfo=UTC)


def ensemble(*scores: float):
    """Make reproducible synthetic evidence with independent source groups."""
    evidence = tuple(
        EnsembleEvidence(
            source_id=f"model-{index}",
            source_version="1.0.0",
            observed_at=AS_OF - timedelta(minutes=2),
            horizon="1D",
            raw_score=score,
            normalization="bounded",
            confidence=1.0,
            reliability=1.0,
            uncertainty=0.0,
            correlation_group=f"group-{index}",
            rationale=f"Synthetic test contribution {index}.",
        )
        for index, score in enumerate(scores, start=1)
    )
    return combine_evidence(
        EnsembleRequest(
            asset_symbol="SYNTH",
            as_of=AS_OF,
            horizon="1D",
            data_snapshot_id="snapshot-synthetic-v1",
            evidence=evidence,
        )
    )


def request(
    result=None,
    *,
    expected_return: float = 0.04,
    expected_risk: float = 0.10,
    forecast_uncertainty: float = 0.10,
    forecast_horizon: str = "1D",
    forecast_observed_at: datetime = AS_OF - timedelta(minutes=1),
    constraints: tuple[DecisionConstraint, ...] = (),
) -> DecisionRequest:
    """Create a request with explicitly sourced, horizon-matched synthetic forecast."""
    return DecisionRequest(
        ensemble=result or ensemble(0.8, 0.7),
        forecast=Forecast(
            horizon=forecast_horizon,
            expected_return=expected_return,
            expected_risk=expected_risk,
            uncertainty=forecast_uncertainty,
        ),
        forecast_source_id="forecast-synthetic",
        forecast_source_version="1.0.0",
        forecast_observed_at=forecast_observed_at,
        regime="SYNTHETIC",
        model_versions=("model-a@1.0.0",),
        feature_versions=("features-a@1.0.0",),
        constraints=constraints,
    )


def calibrated_config(**overrides) -> DecisionConfig:
    """Return policy metadata that explicitly references a prior OOS calibration."""
    values = {
        "probability_calibration_status": "validated_oos",
        "calibration_id": "cal-synthetic-v1",
        "calibration_fit_fingerprint": "a" * 64,
        "calibration_version": "1.0.0",
        "calibration_data_snapshot_id": "calibration-snapshot-v1",
        "calibration_window_start": datetime(2026, 8, 1, tzinfo=UTC),
        "calibration_window_end": datetime(2026, 10, 20, tzinfo=UTC),
        "calibration_validation_data_snapshot_id": "calibration-validation-snapshot-v1",
        "calibration_validation_window_start": datetime(2026, 10, 21, tzinfo=UTC),
        "calibration_validation_window_end": datetime(2026, 10, 22, tzinfo=UTC),
        "calibration_validation_log_loss": 0.7,
        "calibration_validation_baseline_log_loss": 1.0,
    }
    values.update(overrides)
    return DecisionConfig(**values)


def test_calibrated_bullish_evidence_produces_buy_and_preserves_audit() -> None:
    result = make_decision(request(), calibrated_config())

    assert result.status == "DECIDED"
    assert result.record.signal == SignalType.BUY
    assert result.record.recommended_horizon == "1D"
    assert result.record.suggested_position_size == 0.0
    assert len(result.evidence) == 2
    assert result.agreement == pytest.approx(1.0)
    assert (
        result.probabilities.buy + result.probabilities.hold + result.probabilities.sell
    ) == pytest.approx(1.0)
    assert result.record.signal_probability == pytest.approx(result.probabilities.buy)
    assert result.ensemble_fingerprint == request().ensemble.input_fingerprint
    assert len(result.decision_fingerprint) == 64


def test_calibrated_bearish_evidence_produces_sell_when_forecast_agrees() -> None:
    result = make_decision(
        request(ensemble(-0.8, -0.7), expected_return=-0.04),
        calibrated_config(),
    )

    assert result.status == "DECIDED"
    assert result.record.signal == SignalType.SELL
    assert result.record.signal_probability == pytest.approx(result.probabilities.sell)


def test_default_policy_abstains_on_uncalibrated_probabilities() -> None:
    result = make_decision(request())

    assert result.status == "ABSTAINED"
    assert result.abstention_reason == "probabilities_not_validated_out_of_sample"
    assert result.record.signal == SignalType.HOLD
    assert result.record.model_confidence == 0.0
    assert result.record.suggested_position_size == 0.0


def test_midrange_evidence_produces_hold_when_preconditions_pass() -> None:
    result = make_decision(
        request(ensemble(0.05, 0.02)),
        calibrated_config(),
    )

    assert result.status == "DECIDED"
    assert result.record.signal == SignalType.HOLD
    assert result.record.signal_probability == pytest.approx(result.probabilities.hold)


def test_f34_abstention_propagates_to_non_actionable_hold() -> None:
    result = make_decision(
        request(ensemble(0.8)),
        calibrated_config(),
    )

    assert result.status == "ABSTAINED"
    assert result.abstention_reason is not None
    assert result.abstention_reason.startswith("ensemble_abstained:")
    assert result.record.signal == SignalType.HOLD


def test_failed_blocking_constraint_forces_abstention() -> None:
    result = make_decision(
        request(
            constraints=(
                DecisionConstraint(
                    constraint_id="data-quality",
                    satisfied=False,
                    blocking=True,
                    reason="Required data quality check did not pass.",
                ),
            )
        ),
        calibrated_config(),
    )

    assert result.status == "ABSTAINED"
    assert result.abstention_reason == "blocking_constraints_failed:data-quality"
    assert result.constraints[0].constraint_id == "data-quality"


def test_failed_nonblocking_constraint_is_audited_without_veto() -> None:
    result = make_decision(
        request(
            constraints=(
                DecisionConstraint(
                    constraint_id="optional-diagnostic",
                    satisfied=False,
                    blocking=False,
                    reason="Optional diagnostic is unavailable.",
                ),
            )
        ),
        calibrated_config(),
    )

    assert result.status == "DECIDED"
    assert result.record.signal == SignalType.BUY
    assert result.constraints[0].satisfied is False


def test_conflicting_directional_evidence_is_rejected_by_agreement_gate() -> None:
    result = make_decision(
        request(ensemble(1.0, -1.0)),
        calibrated_config(),
    )

    assert result.status == "ABSTAINED"
    assert result.abstention_reason == "agreement_below_minimum"


def test_forecast_uncertainty_is_combined_with_ensemble_uncertainty() -> None:
    result = make_decision(
        request(forecast_uncertainty=1.0),
        calibrated_config(),
    )

    assert result.status == "ABSTAINED"
    assert result.uncertainty == pytest.approx(1.0)
    assert result.abstention_reason == "uncertainty_above_maximum"


def test_forecast_must_match_ensemble_horizon() -> None:
    with pytest.raises(ValidationError, match="forecast horizon must match"):
        request(forecast_horizon="1W")


def test_future_forecast_is_rejected_before_decisioning() -> None:
    with pytest.raises(ValidationError, match="must not be later than ensemble as_of"):
        request(forecast_observed_at=AS_OF + timedelta(seconds=1))


def test_calibration_window_cannot_leak_future_information() -> None:
    config = calibrated_config(
        calibration_window_start=datetime(2026, 10, 24, tzinfo=UTC),
        calibration_window_end=datetime(2026, 10, 25, tzinfo=UTC),
        calibration_validation_window_start=datetime(2026, 10, 26, tzinfo=UTC),
        calibration_validation_window_end=datetime(2026, 10, 27, tzinfo=UTC),
    )
    result = make_decision(request(), config)

    assert result.status == "ABSTAINED"
    assert result.abstention_reason == "calibration_window_after_decision_as_of"


def test_calibration_validation_window_cannot_leak_future_information() -> None:
    config = calibrated_config(
        calibration_validation_window_start=datetime(2026, 10, 24, tzinfo=UTC),
        calibration_validation_window_end=datetime(2026, 10, 25, tzinfo=UTC),
    )
    result = make_decision(request(), config)

    assert result.status == "ABSTAINED"
    assert result.abstention_reason == "calibration_validation_after_decision_as_of"


def test_decision_fingerprint_is_reproducible() -> None:
    decision_request = request()
    policy = calibrated_config()
    first = make_decision(decision_request, policy)
    second = make_decision(decision_request, policy)

    assert first.decision_fingerprint == second.decision_fingerprint
    assert first.record.decision_id == second.record.decision_id
    assert first.record == second.record


def test_calibrated_policy_requires_calibration_provenance() -> None:
    with pytest.raises(ValidationError, match="requires fit and validation provenance"):
        DecisionConfig(probability_calibration_status="validated_oos")


def test_risk_adjusted_score_uses_a_finite_floor_for_zero_risk() -> None:
    result = make_decision(
        request(expected_return=0.01, expected_risk=0.0),
        calibrated_config(),
    )

    assert result.record.risk_adjusted_score == pytest.approx(10_000.0)
