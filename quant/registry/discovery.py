"""Convenience discovery API backed by the curated registry manifest."""
from __future__ import annotations
from .builtins import builtin_algorithms
from .contracts import AlgorithmLifecycle, AlgorithmMetadata
from .registry import AlgorithmRegistry

DEFAULT_REGISTRY = AlgorithmRegistry(builtin_algorithms())

def discover_algorithms(
    *, category: str | None = None, lifecycle: AlgorithmLifecycle | None = None,
    input_name: str | None = None, output_name: str | None = None,
    tag: str | None = None, include_deprecated: bool = False,
) -> tuple[AlgorithmMetadata, ...]:
    """Search known algorithm metadata by category, ports, lifecycle, or tag."""
    return DEFAULT_REGISTRY.discover(
        category=category, lifecycle=lifecycle, input_name=input_name,
        output_name=output_name, tag=tag, include_deprecated=include_deprecated,
    )

__all__ = ["DEFAULT_REGISTRY", "discover_algorithms"]
