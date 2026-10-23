"""F33: metadata registry and discovery for quantitative algorithms."""
from .builtins import BUILTIN_ALGORITHMS, builtin_algorithms
from .contracts import AlgorithmLifecycle, AlgorithmMetadata, AlgorithmParameter, AlgorithmPort, AlgorithmTestStatus
from .discovery import DEFAULT_REGISTRY, discover_algorithms
from .registry import AlgorithmNotFoundError, AlgorithmRegistry, AlgorithmRegistryError

__all__ = [
    "AlgorithmLifecycle", "AlgorithmMetadata", "AlgorithmNotFoundError", "AlgorithmParameter",
    "AlgorithmPort", "AlgorithmRegistry", "AlgorithmRegistryError", "AlgorithmTestStatus",
    "BUILTIN_ALGORITHMS", "DEFAULT_REGISTRY", "builtin_algorithms", "discover_algorithms",
]
