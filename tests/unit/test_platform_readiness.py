"""Behavioral tests for F43 platform readiness aggregation."""

import pytest

from apps.platform.readiness import ComponentCheck, ReadinessStatus, build_readiness

REQUIRED = ("data_quality", "quant_research", "model_evaluation", "decision_engine", "portfolio", "risk_governor", "backtest", "paper_execution", "audit_integrity")


def healthy_checks() -> list[ComponentCheck]:
    return [ComponentCheck(name, True, evidence_id=f"test:{name}") for name in REQUIRED]


def test_all_required_checks_pass_yields_paper_ready_only() -> None:
    result = build_readiness(healthy_checks())
    assert result.status is ReadinessStatus.READY
    assert result.can_paper_trade
    assert not result.can_live_trade
    assert result.execution_mode == "PAPER_ONLY"
    assert result.missing_components == ()


def test_missing_required_component_fails_closed() -> None:
    result = build_readiness([c for c in healthy_checks() if c.component != "risk_governor"])
    assert result.status is ReadinessStatus.BLOCKED
    assert not result.can_paper_trade
    assert result.missing_components == ("risk_governor",)


def test_failed_critical_check_blocks() -> None:
    checks = [ComponentCheck(c.component, c.component != "audit_integrity") for c in healthy_checks()]
    result = build_readiness(checks)
    assert result.status is ReadinessStatus.BLOCKED
    assert result.failed_critical_components == ("audit_integrity",)


def test_failed_optional_check_degrades() -> None:
    result = build_readiness(healthy_checks() + [ComponentCheck("dashboard", False, critical=False)])
    assert result.status is ReadinessStatus.DEGRADED
    assert not result.can_paper_trade


def test_duplicate_component_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_readiness(healthy_checks() + [ComponentCheck("risk_governor", True)])


def test_fingerprint_is_order_independent_and_content_sensitive() -> None:
    checks = healthy_checks()
    assert build_readiness(checks).fingerprint == build_readiness(reversed(checks)).fingerprint
    changed = [ComponentCheck(c.component, c.passed, detail="changed") if c.component == "data_quality" else c for c in checks]
    assert build_readiness(checks).fingerprint != build_readiness(changed).fingerprint


def test_component_contract_rejects_invalid_identifiers() -> None:
    with pytest.raises(ValueError):
        ComponentCheck(" risk_governor", True)
