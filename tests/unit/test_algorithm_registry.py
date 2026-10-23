"""Behavioral tests for the F33 algorithm registry."""

from __future__ import annotations

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
        lifecycle=lifecycle,
        test_status=AlgorithmTestStatus.PASSING,
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


def test_builtin_catalog_is_discoverable_and_does_not_claim_validation() -> None:
    entries = discover_algorithms()
    assert len(entries) >= 10
    assert any(item.algorithm_id == "technical.rsi" for item in entries)
    assert all(item.test_status == AlgorithmTestStatus.NOT_RUN for item in entries)
