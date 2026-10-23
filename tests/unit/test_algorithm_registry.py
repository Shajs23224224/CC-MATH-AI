"""Behavioral tests for the F33 algorithm registry."""

from __future__ import annotations

from importlib import import_module

import pytest
from pydantic import ValidationError

from quant.registry import (
    AlgorithmLifecycle,
    AlgorithmMetadata,
    AlgorithmNotFoundError,
    AlgorithmPort,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    AlgorithmTestStatus,
    discover_algorithms,
)


def _metadata(
    algorithm_id: str = "test.mean",
    version: str = "1.0.0",
    *,
    category: str = "statistics",
    inputs: tuple[AlgorithmPort, ...] = (),
    outputs: tuple[AlgorithmPort, ...] | None = None,
    lifecycle: AlgorithmLifecycle = AlgorithmLifecycle.IMPLEMENTED,
    implementation_ref: str | None = "tests.fixtures.registry_test_algorithm",
    planned_implementation_ref: str | None = None,
    test_status: AlgorithmTestStatus = AlgorithmTestStatus.PASSING,
) -> AlgorithmMetadata:
    return AlgorithmMetadata(
        algorithm_id=algorithm_id,
        name="Test algorithm",
        category=category,
        version=version,
        summary="A registry fixture.",
        method="Deterministic test-only calculation.",
        inputs=inputs,
        outputs=outputs
        or (AlgorithmPort(name="result", dtype="float", description="Numeric result"),),
        limitations=("Synthetic test fixture; not for production decisions.",),
        implementation_ref=implementation_ref,
        planned_implementation_ref=planned_implementation_ref,
        lifecycle=lifecycle,
        test_status=test_status,
    )


def test_metadata_rejects_invalid_id_version_and_duplicate_ports() -> None:
    with pytest.raises(ValidationError, match="lowercase slug"):
        _metadata(algorithm_id="Invalid ID")
    with pytest.raises(ValidationError, match="semantic versioning"):
        _metadata(version="latest")
    with pytest.raises(ValidationError, match="unique"):
        _metadata(
            inputs=(
                AlgorithmPort(name="x", dtype="float", description="x"),
                AlgorithmPort(name="x", dtype="float", description="duplicate x"),
            )
        )


def test_lifecycle_requires_verified_references_and_validation_evidence() -> None:
    with pytest.raises(ValidationError, match="verified implementation_ref"):
        _metadata(implementation_ref=None)

    with pytest.raises(ValidationError, match="passing test evidence"):
        _metadata(lifecycle=AlgorithmLifecycle.VALIDATED, test_status=AlgorithmTestStatus.NOT_RUN)

    planned = _metadata(
        "planned.algorithm",
        lifecycle=AlgorithmLifecycle.EXPERIMENTAL,
        implementation_ref=None,
        planned_implementation_ref="quant.future_module.future_algorithm",
    )
    assert planned.implementation_ref is None
    assert planned.planned_implementation_ref == "quant.future_module.future_algorithm"


def test_register_get_exact_and_latest_version() -> None:
    registry = AlgorithmRegistry((_metadata(version="1.0.0"), _metadata(version="1.9.0")))
    assert registry.get("test.mean").version == "1.9.0"
    assert registry.get("test.mean", "1.0.0").version == "1.0.0"


def test_duplicate_registration_rejected_unless_replacement_explicit() -> None:
    registry = AlgorithmRegistry((_metadata(),))
    with pytest.raises(AlgorithmRegistryError, match="already registered"):
        registry.register(_metadata())
    registry.register(_metadata(), replace=True)
    assert len(registry) == 1


def test_batch_registration_is_atomic_and_rejects_duplicates() -> None:
    registry = AlgorithmRegistry((_metadata("existing.one"),))
    with pytest.raises(AlgorithmRegistryError, match="already registered"):
        registry.register_many((_metadata("new.one"), _metadata("existing.one")))
    assert len(registry) == 1
    with pytest.raises(AlgorithmRegistryError, match="duplicate"):
        registry.register_many((_metadata("duplicate.one"), _metadata("duplicate.one")))
    assert len(registry) == 1


def test_discovery_filters_category_status_ports_and_tags() -> None:
    registry = AlgorithmRegistry(
        (
            _metadata(
                "stats.mean",
                category="statistics",
                inputs=(
                    AlgorithmPort(name="values", dtype="array[float]", description="Observations"),
                ),
            ),
            _metadata("risk.vol", category="risk", lifecycle=AlgorithmLifecycle.EXPERIMENTAL),
            _metadata("risk.old", category="risk", lifecycle=AlgorithmLifecycle.DEPRECATED),
        )
    )
    assert [item.algorithm_id for item in registry.discover(category="statistics")] == [
        "stats.mean"
    ]
    assert [
        item.algorithm_id for item in registry.discover(lifecycle=AlgorithmLifecycle.EXPERIMENTAL)
    ] == ["risk.vol"]
    assert registry.discover(input_name="values")[0].algorithm_id == "stats.mean"
    assert all(item.lifecycle != AlgorithmLifecycle.DEPRECATED for item in registry.discover())
    assert len(registry.discover(include_deprecated=True)) == 3


def test_compatibility_checks_required_inputs_types_and_units() -> None:
    source = _metadata(
        "source.prices",
        outputs=(
            AlgorithmPort(
                name="returns", dtype="series[float]", description="Returns", unit="ratio"
            ),
        ),
    )
    target = _metadata(
        "target.risk",
        inputs=(
            AlgorithmPort(
                name="returns", dtype="series[float]", description="Returns", unit="currency"
            ),
        ),
    )
    errors = AlgorithmRegistry().validate_compatibility(source, target)
    assert len(errors) == 1
    assert "unit" in errors[0]


def test_unknown_algorithm_raises_domain_error() -> None:
    with pytest.raises(AlgorithmNotFoundError):
        AlgorithmRegistry().get("not.registered")


def test_builtin_implemented_references_resolve_to_public_callables() -> None:
    entries = discover_algorithms()
    implemented = [item for item in entries if item.lifecycle == AlgorithmLifecycle.IMPLEMENTED]

    assert implemented
    for item in implemented:
        assert item.implementation_ref is not None
        module_name, symbol_name = item.implementation_ref.rsplit(".", maxsplit=1)
        module = import_module(module_name)
        assert callable(getattr(module, symbol_name, None)), item.algorithm_id


def test_builtin_catalog_is_discoverable_and_does_not_claim_validation() -> None:
    entries = discover_algorithms()
    identifiers = [item.algorithm_id for item in entries]
    assert len(entries) >= 300
    assert len(identifiers) == len(set(identifiers))
    assert any(item.algorithm_id == "technical.rsi" for item in entries)
    assert all(item.test_status == AlgorithmTestStatus.NOT_RUN for item in entries)

    implemented = [item for item in entries if item.lifecycle == AlgorithmLifecycle.IMPLEMENTED]
    planned = [item for item in entries if item.planned_implementation_ref is not None]
    assert implemented
    assert all(item.implementation_ref is not None for item in implemented)
    assert len(planned) >= 250
    assert all(item.lifecycle == AlgorithmLifecycle.EXPERIMENTAL for item in planned)
    assert all(item.implementation_ref is None for item in planned)


def test_f36_portfolio_optimizers_are_registered_without_claiming_validation() -> None:
    expected_ids = {
        "portfolio.equal_weight",
        "portfolio.mean_variance",
        "portfolio.minimum_variance",
        "portfolio.maximum_sharpe",
        "portfolio.risk_parity",
        "portfolio.inverse_volatility",
        "portfolio.hrp",
        "portfolio.black_litterman",
        "portfolio.maximum_diversification",
    }
    portfolio_entries = discover_algorithms(category="portfolio")
    entries_by_id = {item.algorithm_id: item for item in portfolio_entries}

    assert expected_ids.issubset(entries_by_id)
    for algorithm_id in expected_ids:
        item = entries_by_id[algorithm_id]
        assert item.implementation_ref == "portfolio.optimize_portfolio"
        assert item.lifecycle == AlgorithmLifecycle.IMPLEMENTED
        assert item.test_status == AlgorithmTestStatus.NOT_RUN
        assert "portfolio-optimization" in item.tags


def test_f37_position_sizing_methods_are_registered_without_claiming_validation() -> None:
    expected_ids = {
        "portfolio.fixed_fractional",
        "portfolio.volatility_based",
        "portfolio.risk_based",
        "portfolio.kelly",
        "portfolio.fractional_kelly",
        "portfolio.confidence_weighting",
    }
    entries_by_id = {item.algorithm_id: item for item in discover_algorithms(category="portfolio")}

    assert expected_ids.issubset(entries_by_id)
    for algorithm_id in expected_ids:
        item = entries_by_id[algorithm_id]
        assert item.implementation_ref == "portfolio.size_positions"
        assert item.lifecycle == AlgorithmLifecycle.IMPLEMENTED
        assert item.test_status == AlgorithmTestStatus.NOT_RUN
        assert "F37" in item.tags


def test_f38_portfolio_risk_governor_is_registered_without_claiming_validation() -> None:
    entries_by_id = {item.algorithm_id: item for item in discover_algorithms(category="risk")}

    item = entries_by_id["risk.portfolio_governor"]
    assert item.implementation_ref == "risk.evaluate_risk_governor"
    assert item.lifecycle == AlgorithmLifecycle.IMPLEMENTED
    assert item.test_status == AlgorithmTestStatus.NOT_RUN
    assert "F38" in item.tags
