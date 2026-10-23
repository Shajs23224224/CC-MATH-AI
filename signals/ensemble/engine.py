"""Deterministic, auditable aggregation for quantitative evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

from .contracts import (
    EnsembleConfig,
    EnsembleContribution,
    EnsembleEvidence,
    EnsembleRequest,
    EnsembleResult,
    ExcludedEvidence,
)


def combine_evidence(
    request: EnsembleRequest,
    config: EnsembleConfig | None = None,
) -> EnsembleResult:
    """Combine source evidence while capping correlated-source influence.

    Raw numeric values are normalized by the method declared on each evidence record.
    Confidence and reliability attenuate weights. For each correlation group, total
    effective weight is capped at that group's largest source weight, so adding several
    highly correlated models cannot linearly multiply the group's influence. The result
    is research evidence only; it does not issue decisions, orders, or risk approvals.
    """
    policy = config or EnsembleConfig()
    explicit_weights = {item.source_id: item.weight for item in policy.source_weights}
    excluded: list[ExcludedEvidence] = []
    candidates: list[tuple[EnsembleEvidence, float, str]] = []

    for item in request.evidence:
        if not item.eligible:
            excluded.append(
                ExcludedEvidence(
                    source_id=item.source_id,
                    source_version=item.source_version,
                    reason=item.exclusion_reason or "upstream marked evidence ineligible",
                )
            )
            continue

        configured_weight = explicit_weights.get(item.source_id, policy.default_weight)
        if configured_weight == 0.0:
            reason = "source weight is zero in the ensemble configuration"
            excluded.append(
                ExcludedEvidence(
                    source_id=item.source_id, source_version=item.source_version, reason=reason
                )
            )
            continue

        effective_weight = configured_weight * item.confidence * item.reliability
        if effective_weight <= 0.0:
            reason = (
                "zero confidence"
                if item.confidence == 0.0
                else "zero reliability"
                if item.reliability == 0.0
                else "effective weight is zero"
            )
            excluded.append(
                ExcludedEvidence(
                    source_id=item.source_id, source_version=item.source_version, reason=reason
                )
            )
            continue

        group = item.correlation_group or item.source_id
        candidates.append((item, effective_weight, group))

    group_members: dict[str, list[tuple[EnsembleEvidence, float, str]]] = defaultdict(list)
    for candidate in candidates:
        group_members[candidate[2]].append(candidate)

    adjusted: list[tuple[EnsembleEvidence, float, str]] = []
    for group in sorted(group_members):
        members = group_members[group]
        raw_total = math.fsum(item[1] for item in members)
        group_cap = max(item[1] for item in members)
        for item, raw_weight, source_group in members:
            adjusted_weight = group_cap * raw_weight / raw_total
            adjusted.append((item, adjusted_weight, source_group))

    total_weight = math.fsum(item[1] for item in adjusted)
    contributions: list[EnsembleContribution] = []
    normalized_values: list[tuple[EnsembleEvidence, float, float]] = []

    if total_weight > 0.0:
        for item, adjusted_weight, group in adjusted:
            normalized_weight = adjusted_weight / total_weight
            normalized_score = item.normalized_score
            contribution = normalized_weight * normalized_score
            contributions.append(
                EnsembleContribution(
                    source_id=item.source_id,
                    source_version=item.source_version,
                    correlation_group=group,
                    normalized_score=normalized_score,
                    effective_weight=adjusted_weight,
                    normalized_weight=normalized_weight,
                    contribution=contribution,
                    rationale=item.rationale,
                    observed_at=item.observed_at,
                )
            )
            normalized_values.append((item, normalized_score, normalized_weight))

    contributions.sort(key=lambda item: (item.source_id, item.source_version))
    excluded.sort(key=lambda item: (item.source_id, item.source_version, item.reason))
    independent_group_count = len(group_members)
    source_count = len(contributions)

    if source_count == 0:
        score = 0.0
        agreement = 0.0
        uncertainty = 1.0
        abstained = True
        abstention_reason = "no_eligible_evidence"
    else:
        score = max(
            -1.0,
            min(1.0, math.fsum(item.contribution for item in contributions)),
        )
        positive_weight = math.fsum(
            weight for _, value, weight in normalized_values if value > policy.neutral_threshold
        )
        negative_weight = math.fsum(
            weight for _, value, weight in normalized_values if value < -policy.neutral_threshold
        )
        neutral_weight = max(0.0, 1.0 - positive_weight - negative_weight)
        agreement = min(1.0, neutral_weight + max(positive_weight, negative_weight))

        mean_source_uncertainty = math.fsum(
            weight * (1.0 - (1.0 - item.uncertainty) * item.confidence * item.reliability)
            for item, _, weight in normalized_values
        )
        dispersion = math.fsum(
            weight * abs(value - score) / 2.0
            for _, value, weight in normalized_values
        )
        uncertainty = min(
            1.0, max(0.0, 1.0 - (1.0 - mean_source_uncertainty) * (1.0 - dispersion))
        )
        if independent_group_count < policy.min_independent_groups:
            abstained = True
            abstention_reason = (
                f"requires at least {policy.min_independent_groups} independent groups; "
                f"received {independent_group_count}"
            )
            uncertainty = 1.0
        else:
            abstained = False
            abstention_reason = None

    fingerprint = _fingerprint(request, policy)
    return EnsembleResult(
        asset_symbol=request.asset_symbol,
        as_of=request.as_of,
        horizon=request.horizon,
        data_snapshot_id=request.data_snapshot_id,
        config_id=policy.config_id,
        config_version=policy.version,
        score=score,
        agreement=agreement,
        uncertainty=uncertainty,
        abstained=abstained,
        abstention_reason=abstention_reason,
        source_count=source_count,
        independent_group_count=independent_group_count,
        contributions=tuple(contributions),
        excluded_evidence=tuple(excluded),
        input_fingerprint=fingerprint,
    )


def _fingerprint(request: EnsembleRequest, config: EnsembleConfig) -> str:
    """Fingerprint canonical request evidence and policy for exact replay metadata."""
    request_payload = request.model_dump(mode="json")
    request_payload["evidence"] = sorted(
        request_payload["evidence"],
        key=lambda item: (item["source_id"], item["source_version"]),
    )
    config_payload = config.model_dump(mode="json")
    config_payload["source_weights"] = sorted(
        config_payload["source_weights"], key=lambda item: item["source_id"]
    )
    canonical = json.dumps(
        {"request": request_payload, "config": config_payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["combine_evidence"]
