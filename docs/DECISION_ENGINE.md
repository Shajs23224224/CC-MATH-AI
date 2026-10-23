# F35 — Meta-model / Decision Engine

## Purpose

F35 converts a F34 ensemble result and an independently supplied, horizon-matched forecast into a deterministic, auditable decision output. Its public entry point is `make_decision` from `signals.decision`.

The output contains the legacy core `DecisionRecord` plus BUY/HOLD/SELL class probabilities, status, abstention reason, agreement, combined uncertainty, source-level evidence, pre-decision constraints, configuration identity, calibration provenance and SHA-256 fingerprints.

F35 never places orders, proposes a position size, optimizes a portfolio or grants risk approval. The record deliberately sets `suggested_position_size=0`; F36/F37 handle allocation and sizing, and downstream risk controls remain authoritative. An abstained record uses HOLD only as a non-actionable placeholder; consumers must check `status` / `abstained`.

## Inputs and provenance

A `DecisionRequest` requires:

- The F34 `EnsembleResult`, including its as-of timestamp, horizon, data snapshot, score, agreement, uncertainty, abstention status, contributions and fingerprint.
- A `Forecast` with the exact same horizon. The forecast source ID/version and timezone-aware `forecast_observed_at` must be supplied.
- Optional regime, model/feature versions and typed pre-decision constraints.
- Forecast observation time must not be later than the F34 as-of time. Missing horizons, future forecast timestamps, non-finite forecasts and duplicate constraint IDs are rejected at validation time.

Expected return and expected risk are not inferred from the dimensionless F34 score. They come from the supplied forecast and must use compatible units and the same horizon. The risk-adjusted score is `expected_return / max(expected_risk, minimum_risk_floor)`; it is a bounded-denominator return-to-risk ratio, not a Sharpe ratio or an estimate of expected utility.

Pre-decision constraints are eligibility checks such as data or model readiness. A blocking failure abstains the decision. These checks are not a substitute for F21 risk metrics or the future Portfolio Risk Governor.

## Deterministic decision policy

The probability transform uses a numerically stable three-class softmax:

- BUY logit: `ensemble.score / probability_temperature`
- HOLD logit: `hold_bias`
- SELL logit: `-ensemble.score / probability_temperature`

The temperature and HOLD bias are versioned policy values. By themselves they produce model-derived probabilities, not empirically calibrated probabilities. The default policy therefore requires an explicitly validated calibration artifact and abstains when none is configured.

A BUY/SELL decision requires a directional ensemble score beyond its configured threshold, enough class probability, sufficient agreement, uncertainty below the configured ceiling, no blocking constraint failures and an expected-return sign aligned with the directional signal unless that alignment rule is explicitly disabled. Scores between the directional thresholds produce HOLD when all acceptance gates pass.

Decision confidence is the selected class probability multiplied by ensemble agreement and one minus combined uncertainty. Forecast uncertainty is normalized by the explicitly versioned `forecast_uncertainty_scale`, capped at one, then combined with ensemble uncertainty as:

`1 - (1 - ensemble_uncertainty) × (1 - normalized_forecast_uncertainty)`

This is an engineering confidence diagnostic; it is not a guarantee that the decision will be correct.

## Probability calibration and temporal leakage controls

Use `fit_probability_calibration` only with scores generated out of sample and labels produced by a separately defined, versioned outcome policy. Every calibration sample records its forecast-origin time, the time its label became observable, horizon and data-snapshot ID.

The fitter requires at least 30 total samples by default, a single horizon and at least two examples of each class in both temporal slices. Samples are ordered by forecast-origin time. The early 70% is the fit candidate set; any training sample whose label was not available by the beginning of the later validation slice is purged. Temperature and HOLD bias are chosen by deterministic grid search minimizing multiclass log loss on the retained training slice. The later slice is not used to fit parameters; its log loss is compared against the baseline policy.

The result reports separate training and validation counts, fit/validation time windows, calibration and validation data-snapshot IDs, fitted parameters, train loss, validation loss, baseline validation loss, whether validation passed and a fingerprint of the calibration inputs. A policy marked `validated_oos` must carry these fields, reference the fit fingerprint and show validation log loss no worse than baseline; validation and fit windows must be chronological and no later than decision time.

This is a guarded engineering calibration step, not proof of probabilistic calibration or live predictive skill. It cannot verify the truth of the supplied labels, the quality of the underlying forecasts, data-selection bias or an external claim that an artifact came from this function. Keep the artifact and snapshot IDs in immutable experiment storage, and require a fresh, later validation window when the policy is retuned.

## Example of creating a policy from a calibration artifact

The sample collection is omitted here because it must come from the project's persisted, point-in-time-safe OOS evaluation dataset.

    calibration = fit_probability_calibration(
        calibration_samples,
        calibration_id="decision-temperature-v1",
        calibration_version="1.0.0",
        calibration_data_snapshot_id="calibration-fit-dataset-v1",
        calibration_validation_data_snapshot_id="calibration-validation-dataset-v1",
        fit_as_of=fit_as_of,
    )

    if calibration.validation_passed:
        policy = DecisionConfig(
            config_id="decision-policy",
            version="1.0.0",
            probability_temperature=calibration.temperature,
            hold_bias=calibration.hold_bias,
            probability_calibration_status="validated_oos",
            calibration_id=calibration.calibration_id,
            calibration_fit_fingerprint=calibration.fit_fingerprint,
            calibration_version=calibration.calibration_version,
            calibration_data_snapshot_id=calibration.calibration_data_snapshot_id,
            calibration_window_start=calibration.calibration_window_start,
            calibration_window_end=calibration.calibration_window_end,
            calibration_validation_data_snapshot_id=calibration.calibration_validation_data_snapshot_id,
            calibration_validation_window_start=calibration.validation_window_start,
            calibration_validation_window_end=calibration.validation_window_end,
            calibration_validation_log_loss=calibration.calibrated_log_loss,
            calibration_validation_baseline_log_loss=calibration.baseline_log_loss,
        )

The caller should keep the default abstaining policy when the temporal calibration validation fails. A passed temporal split does not establish profitability; later walk-forward and adversarial validation remain separate phases.

## Abstention reasons

The engine rejects or abstains on:

- F34 abstention/no eligible independent evidence.
- Uncalibrated probabilities when the policy requires a validated artifact.
- Calibration or validation data that extend beyond the decision as-of time.
- Failed blocking constraints.
- Agreement below minimum or combined uncertainty above maximum.
- BUY/SELL probabilities below the action threshold.
- Conflict between a directional ensemble signal and the expected-return sign.
- Non-finite risk-adjusted numerics.

Every abstention preserves its reason and returns a non-actionable HOLD record with zero model confidence and zero suggested position size. Downstream consumers must not convert that placeholder into an order.

## Validation scope and limitations

The F35 workflow runs on Python 3.11 and 3.12 and includes Ruff, format checks, mypy, behavioral tests and regressions for F34/F33, F32–F28. Synthetic tests verify software invariants only. The engine does not execute trades, train the upstream forecasting models, determine realized-label definitions, guarantee future calibration, or replace Risk Governor, walk-forward, Monte Carlo or paper-trading controls.
