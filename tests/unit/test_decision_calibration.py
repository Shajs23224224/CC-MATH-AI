"""Tests for F35 out-of-sample probability calibration and leakage controls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.contracts import SignalType
from signals.decision import DecisionCalibrationSample, fit_probability_calibration

AS_OF = datetime(2026, 10, 23, 15, 0, tzinfo=UTC)


def samples() -> tuple[DecisionCalibrationSample, ...]:
    """Construct 36 synthetic, matured, single-horizon calibration outcomes."""
    examples: list[tuple[float, SignalType]] = []
    for _ in range(12):
        examples.append((0.65, SignalType.BUY))
        examples.append((0.02, SignalType.HOLD))
        examples.append((-0.65, SignalType.SELL))
    return tuple(
        DecisionCalibrationSample(
            sample_id=f"sample-{index:03d}",
            score=score,
            realized_class=label,
            horizon="1D",
            observed_at=AS_OF - timedelta(days=40 - index),
            label_available_at=AS_OF - timedelta(days=39 - index),
            data_snapshot_id=f"forecast-snapshot-{index:03d}",
        )
        for index, (score, label) in enumerate(examples)
    )


def fit(data: tuple[DecisionCalibrationSample, ...]):
    return fit_probability_calibration(
        data,
        calibration_id="calibration-synthetic-v1",
        calibration_version="1.0.0",
        calibration_data_snapshot_id="calibration-dataset-v1",
        calibration_validation_data_snapshot_id="calibration-validation-dataset-v1",
        fit_as_of=AS_OF,
    )


def test_calibrator_fits_reproducible_parameters_and_reports_loss() -> None:
    result = fit(samples())

    assert result.sample_count == 36
    assert result.horizon == "1D"
    assert result.temperature >= 0.05
    assert result.calibrated_log_loss <= result.baseline_log_loss
    assert result.calibration_window_end <= result.fit_as_of
    assert len(result.fit_fingerprint) == 64
    assert result.training_sample_count == 25
    assert result.validation_sample_count == 11
    assert result.purged_sample_count == 0
    assert result.validation_passed
    assert result.calibration_window_end <= result.validation_window_start
    assert result.validation_window_end <= result.fit_as_of
    assert dict(result.class_counts) == {
        SignalType.BUY: 9,
        SignalType.HOLD: 8,
        SignalType.SELL: 8,
    }
    assert dict(result.validation_class_counts) == {
        SignalType.BUY: 3,
        SignalType.HOLD: 4,
        SignalType.SELL: 4,
    }


def test_calibrator_is_invariant_to_input_order() -> None:
    data = samples()

    first = fit(data)
    second = fit(tuple(reversed(data)))

    assert first.fit_fingerprint == second.fit_fingerprint
    assert first.temperature == second.temperature
    assert first.hold_bias == second.hold_bias
    assert first.calibrated_log_loss == second.calibrated_log_loss


def test_calibrator_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 30 calibration samples"):
        fit_probability_calibration(
            samples()[:29],
            calibration_id="calibration-v1",
            calibration_version="1.0.0",
            calibration_data_snapshot_id="dataset-v1",
            calibration_validation_data_snapshot_id="validation-dataset-v1",
            fit_as_of=AS_OF,
        )


def test_calibrator_rejects_labels_not_yet_observable() -> None:
    data = list(samples())
    data[0] = data[0].model_copy(update={"label_available_at": AS_OF + timedelta(seconds=1)})

    with pytest.raises(ValueError, match="label is not yet observable"):
        fit(tuple(data))


def test_calibrator_rejects_mixed_horizons() -> None:
    data = list(samples())
    data[-1] = data[-1].model_copy(update={"horizon": "1W"})

    with pytest.raises(ValueError, match="same horizon"):
        fit(tuple(data))


def test_calibrator_requires_all_three_classes() -> None:
    data = tuple(
        sample.model_copy(update={"realized_class": SignalType.BUY}) for sample in samples()
    )

    with pytest.raises(
        ValueError, match="training slice requires at least two samples for each class"
    ):
        fit(data)


def test_calibrator_rejects_future_forecast_origins() -> None:
    data = list(samples())
    data[0] = data[0].model_copy(
        update={
            "observed_at": AS_OF + timedelta(minutes=1),
            "label_available_at": AS_OF + timedelta(minutes=2),
        }
    )

    with pytest.raises(ValueError, match="forecast observed_at is later"):
        fit(tuple(data))
