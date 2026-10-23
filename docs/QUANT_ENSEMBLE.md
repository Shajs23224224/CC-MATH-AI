# F34 — Quant Ensemble

## Purpose and boundaries

F34 combines heterogeneous quantitative evidence into one reproducible, auditable research score while preserving every accepted source contribution and every exclusion reason. It does not produce the final BUY/HOLD/SELL decision, size a position, place orders, or override the F21 Risk Engine / later Risk Governor. F35 consumes ensemble evidence to build a decision record.

Public API example (the evidence is synthetic and is not an investment recommendation):

    from datetime import datetime, timezone

    from signals.ensemble import EnsembleConfig, EnsembleEvidence, EnsembleRequest, combine_evidence

    request = EnsembleRequest(
        asset_symbol="SYNTH",
        as_of=datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc),
        horizon="1D",
        data_snapshot_id="market-snapshot-v1",
        evidence=(
            EnsembleEvidence(
                source_id="technical.rsi",
                source_version="1.0.0",
                observed_at=datetime(2026, 10, 23, 14, 55, tzinfo=timezone.utc),
                horizon="1D",
                raw_score=0.4,
                normalization="bounded",
                confidence=0.8,
                reliability=0.9,
                uncertainty=0.2,
                correlation_group="price-momentum",
                rationale="Synthetic RSI-derived evidence.",
            ),
            EnsembleEvidence(
                source_id="forecast.model-a",
                source_version="2.1.0",
                observed_at=datetime(2026, 10, 23, 14, 58, tzinfo=timezone.utc),
                horizon="1D",
                raw_score=0.6,
                normalization="bounded",
                confidence=0.9,
                reliability=0.8,
                uncertainty=0.25,
                correlation_group="forecasting",
                rationale="Synthetic out-of-sample forecast evidence.",
            ),
        ),
    )
    result = combine_evidence(request, EnsembleConfig(min_independent_groups=2))

## Normalization

Each input declares one normalization method, with bounded output in [-1, 1]:

- bounded: the source must already provide a score in [-1, 1].
- tanh: tanh(raw_score / normalization_scale).
- zscore: tanh((raw_score - normalization_center) / normalization_scale).

Centers and scales must be calibrated using only information available by the request's as-of time. F34 validates the declared parameters but cannot prove how upstream calibration data were selected.

All input scores must be finite. Timestamps must be timezone-aware, every evidence record must use the request horizon, and no observed_at may be later than as_of. Duplicate source IDs in one request are rejected to prevent accidental double-counting.

## Weighting and correlation control

For each source, its preliminary weight is:

    configured source weight × confidence × reliability

A source-specific weight overrides default_weight; an explicit weight of zero disables the source. Ineligible, disabled, zero-confidence and zero-reliability sources are excluded with a recorded reason.

To limit double-counting, the total effective weight of each correlation group is capped at the largest preliminary source weight in that group. Members divide that capped weight in proportion to their preliminary weights. A source without a declared group forms its own group. This deterministic cap is a conservative engineering heuristic, not a statistical estimate of true dependence.

The final score is the sum of each normalized evidence score multiplied by its normalized contribution weight. Individual contributions must sum to the aggregate score, and normalized weights must sum to one.

## Agreement and uncertainty

Agreement is the non-contradictory directional mass: neutral evidence (within neutral_threshold) plus the greater of total positive and total negative weights. Balanced positive/negative evidence therefore has agreement near 0.5, while unopposed direction or all-neutral evidence approaches 1.0. Interpret agreement together with the score; a high agreement around a zero score may simply indicate neutral inputs.

Per-source uncertainty is combined with confidence and reliability. The engine also computes weighted dispersion around the aggregate score and combines it with source uncertainty. These are deterministic diagnostics, not calibrated probabilities that a forecast will be correct.

## Abstention and output

The result abstains when no source remains eligible or when the number of independent correlation groups is below min_independent_groups. An abstained result has uncertainty 1.0 and an explicit reason; it is not an actionable signal. Where evidence exists but independent groups are insufficient, the computed score remains available for diagnostics but callers must respect abstained=true.

Each EnsembleResult contains the asset, information cutoff, horizon, data snapshot ID, configuration ID/version, score, agreement, uncertainty, abstention state, contributing source versions, effective and normalized weights, individual contributions, exclusions and a SHA-256 input/configuration fingerprint. The fingerprint sorts evidence and source-weight overrides so input ordering does not change replay identity.

## Validation and limitations

The dedicated workflow tests Python 3.11/3.12, lints and formats F33/F34, runs mypy, executes registry/ensemble behavioral tests, and reruns F32/F31/F30/F29 regressions. General CI and Security are separate acceptance gates.

The aggregation is a deterministic weighted mean with correlation-group weight caps; it does not learn weights, calibrate predictive probabilities, infer a correlation matrix or prove causal independence. Metadata status in the F33 registry is never auto-promoted from F34. Risk evaluation, portfolio optimization and trade decisions remain downstream responsibilities.
