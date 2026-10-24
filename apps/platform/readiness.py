"""Deterministic readiness aggregation for the C-MATH-AI Platform 1.0 milestone."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class ReadinessStatus(StrEnum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


_REQUIRED_COMPONENTS = frozenset(
    {
        "data_quality",
        "quant_research",
        "model_evaluation",
        "decision_engine",
        "portfolio",
        "risk_governor",
        "backtest",
        "paper_execution",
        "audit_integrity",
    }
)


@dataclass(frozen=True, slots=True)
class ComponentCheck:
    """Observed component result; passed must reflect an actual check, not configuration."""

    component: str
    passed: bool
    detail: str = ""
    evidence_id: str = ""
    critical: bool = True

    def __post_init__(self) -> None:
        if not self.component or self.component.strip() != self.component:
            raise ValueError("component must be a non-empty, trimmed identifier")
        if not isinstance(self.passed, bool) or not isinstance(self.critical, bool):
            raise TypeError("passed and critical must be booleans")
        if not isinstance(self.detail, str) or not isinstance(self.evidence_id, str):
            raise TypeError("detail and evidence_id must be strings")


@dataclass(frozen=True, slots=True)
class PlatformReadiness:
    """Immutable readiness snapshot with a stable digest and explicit safety boundary."""

    status: ReadinessStatus
    checks: tuple[ComponentCheck, ...]
    missing_components: tuple[str, ...]
    failed_critical_components: tuple[str, ...]
    fingerprint: str
    execution_mode: str = "PAPER_ONLY"

    @property
    def can_paper_trade(self) -> bool:
        return self.status is ReadinessStatus.READY

    @property
    def can_live_trade(self) -> bool:
        """Live execution is intentionally unsupported in F43."""
        return False


def build_readiness(checks: Iterable[ComponentCheck]) -> PlatformReadiness:
    """Aggregate observed checks fail-closed; duplicate component IDs are rejected."""
    ordered = tuple(sorted(checks, key=lambda check: check.component))
    names = [check.component for check in ordered]
    if len(names) != len(set(names)):
        raise ValueError("duplicate component checks are not allowed")

    missing = tuple(sorted(_REQUIRED_COMPONENTS - set(names)))
    failed_critical = tuple(
        check.component
        for check in ordered
        if (check.critical or check.component in _REQUIRED_COMPONENTS)
        and not check.passed
    )
    failed_optional = any(
        not check.passed
        and not check.critical
        and check.component not in _REQUIRED_COMPONENTS
        for check in ordered
    )
    if missing or failed_critical:
        status = ReadinessStatus.BLOCKED
    elif failed_optional:
        status = ReadinessStatus.DEGRADED
    else:
        status = ReadinessStatus.READY

    payload = {
        "status": status.value,
        "checks": [
            {
                "component": check.component,
                "passed": check.passed,
                "detail": check.detail,
                "evidence_id": check.evidence_id,
                "critical": check.critical,
            }
            for check in ordered
        ],
        "missing_components": list(missing),
        "failed_critical_components": list(failed_critical),
        "execution_mode": "PAPER_ONLY",
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return PlatformReadiness(status, ordered, missing, failed_critical, fingerprint)
