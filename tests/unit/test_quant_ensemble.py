"""Behavioral tests for F34 deterministic quantitative ensembling."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from signals.ensemble import (
    EnsembleConfig,
    EnsembleEvidence,
    EnsembleRequest,
    SourceWeight,
    combine_evidence,
)

AS_OF = datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc)


def evidence(
    source_id: str,
    score: float = 0.5,
    *,
    confidence: float = 1.0,
    reliability: float = 1.0,
    uncertainty: float = 0.0,
    group: str | None = None,
    observed_at: datetime = AS_OF,
    horizon: str = "1D",
    normalization: str = "bounded",
    center: float = 0.0,
    scale: float = 1.0,
    eligible: bool = True,
    exclusion_reason: str | None = None,
) -> EnsembleEvidence:
    return EnsembleEvidence(
        source_id=source_id,
        source_version="1.0.0",
        observed_at=observed_at,
        horizon=horizon,
        raw_score=score,
        normalization=normalization,
        normalization_center=center,
        normalization_scale=scale,
        confidence=confidence,
        reliability=reliability,
        uncertainty=uncertainty,
        correlation_group=group,
        eligible=eligible,
        exclusion_reason=exclusion_reason,
        rationale=f"Synthetic evidence from {source_id}.",
    )


def request(*items: EnsembleEvidence, as_of: datetime = AS_OF) -> EnsembleRequest:
    return EnsembleRequest(
        asset_symbol="SYNTH",
        as_of=as_of,
        horizon="1D",
        data_snapshot_id="snapshot-synthetic-001",
        evidence=items,
    )


def test_normalization_methods_are_bounded_and_deterministic() -> None:
    bounded = evidence("bounded", 0.4)
    tanh = evidence("tanh", 3.0, normalization="tanh", scale=1.0)
    zscore = evidence("zscore", 4.0, normalization="zscore", center=2.0, scale=2.0)

    assert bounded.normalized_score == 0.4
    assert tanh.normalized_score == pytest.approx(math.tanh(3.0))
    assert zscore.normalized_score == pytest.approx(math.tanh(1.0))
    assert all(-1.0 <= item.normalized_score <= 1.0 for item in (bounded, tanh, zscore))


def test_equal_weighted_sources_produce_auditable_mean() -> None:
    result = combine_evidence(request(evidence("bull", 1.0), evidence("neutral", 0.0)))

    assert result.score == pytest.approx(0.5)
    assert result.agreement == pytest.approx(1.0)
    assert result.uncertainty == pytest.approx(0.25)
    assert not result.abstained
    assert result.source_count == 2
    assert sum(item.contribution for item in result.contributions) == pytest.approx(result.score)
    assert sum(item.normalized_weight for item in result.contributions) == pytest.approx(1.0)


def test_explicit_source_weights_change_score_reproducibly() -> None:
    policy = EnsembleConfig(
        source_weights=(
            SourceWeight(source_id="bull", weight=3.0),
            SourceWeight(source_id="bear", weight=1.0),
        )
    )
    result = combine_evidence(request(evidence("bull", 1.0), evidence("bear", -1.0)), policy)

    assert result.score == pytest.approx(0.5)
    assert {item.source_id: item.normalized_weight for item in result.contributions} == {
        "bull": pytest.approx(0.75),
        "bear": pytest.approx(0.25),
    }


def test_correlation_group_cap_prevents_duplicate_sources_dominating() -> None:
    result = combine_evidence(
        request(
            evidence("model-a", 1.0, group="same-family"),
            evidence("model-b", 1.0, group="same-family"),
            evidence("independent", -1.0, group="independent-family"),
        )
    )

    assert result.score == pytest.approx(0.0)
    assert result.independent_group_count == 2
    assert result.agreement == pytest.approx(0.5)
    assert sum(
        item.effective_weight
        for item in result.contributions
        if item.correlation_group == "same-family"
    ) == pytest.approx(1.0)


def test_confidence_and_reliability_attenuate_source_weight() -> None:
    result = combine_evidence(
        request(
            evidence("strong", 1.0, confidence=1.0, reliability=1.0),
            evidence("weak", -1.0, confidence=0.25, reliability=1.0),
        )
    )

    assert result.score == pytest.approx(0.6)
    assert result.uncertainty > 0.0


def test_duplicate_source_is_rejected_to_prevent_double_counting() -> None:
    with pytest.raises(ValidationError, match="source_id"):
        request(evidence("same", 0.2), evidence("same", 0.8))


def test_future_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="later than request as_of"):
        request(evidence("future", 0.5, observed_at=AS_OF + timedelta(seconds=1)))


def test_mixed_horizons_are_rejected() -> None:
    with pytest.raises(ValidationError, match="request horizon"):
        request(evidence("daily", 0.2), evidence("weekly", 0.4, horizon="1W"))


def test_same_correlation_group_does_not_satisfy_independence_gate() -> None:
    result = combine_evidence(
        request(
            evidence("a", 0.8, group="one-family"),
            evidence("b", 0.7, group="one-family"),
        )
    )

    assert result.abstained
    assert result.independent_group_count == 1
    assert result.abstention_reason is not None
    assert result.uncertainty == 1.0


def test_ineligible_evidence_is_retained_in_exclusion_audit() -> None:
    policy = EnsembleConfig(min_independent_groups=1)
    result = combine_evidence(
        request(
            evidence("usable", 0.6),
            evidence("stale", -0.9, eligible=False, exclusion_reason="stale source"),
        ),
        policy,
    )

    assert not result.abstained
    assert result.source_count == 1
    assert len(result.excluded_evidence) == 1
    assert result.excluded_evidence[0].reason == "stale source"


def test_zero_confidence_sources_are_excluded_and_cause_safe_abstention() -> None:
    result = combine_evidence(request(evidence("zero", 0.8, confidence=0.0)))

    assert result.abstained
    assert result.score == 0.0
    assert result.uncertainty == 1.0
    assert result.excluded_evidence[0].reason == "zero confidence"


def test_conflicting_sources_raise_uncertainty() -> None:
    result = combine_evidence(request(evidence("bull", 1.0), evidence("bear", -1.0)))

    assert result.score == pytest.approx(0.0)
    assert result.agreement == pytest.approx(0.5)
    assert result.uncertainty == pytest.approx(0.5)


def test_fingerprint_is_invariant_to_input_order() -> None:
    first = combine_evidence(request(evidence("b", -0.3), evidence("a", 0.7)))
    second = combine_evidence(request(evidence("a", 0.7), evidence("b", -0.3)))

    assert first.input_fingerprint == second.input_fingerprint
    assert first.score == second.score
    assert first.contributions == second.contributions


def test_invalid_bounded_score_and_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"bounded raw_score"):
        evidence("invalid", 1.1)
    with pytest.raises(ValidationError, match="timezone-aware"):
        evidence("naive", 0.1, observed_at=datetime(2026, 10, 23, 15, 0))


def test_config_rejects_duplicate_source_weight_overrides() -> None:
    with pytest.raises(ValidationError, match="duplicate source IDs"):
        EnsembleConfig(
            source_weights=(
                SourceWeight(source_id="same", weight=1.0),
                SourceWeight(source_id="same", weight=2.0),
            )
        )


def test_empty_request_abstains_without_fabricating_a_score() -> None:
    result = combine_evidence(request())

    assert result.abstained
    assert result.score == 0.0
    assert result.source_count == 0
    assert result.independent_group_count == 0
    assert len(result.input_fingerprint) == 64
