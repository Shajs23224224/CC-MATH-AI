"""Classical ML, adaptive mathematical search and bounded symbolic discovery."""

from .adaptive import AdaptiveMathEngine, predict_adaptive_candidate
from .classical import ClassicalMLModel, supported_algorithms
from .patterns import discover_patterns
from .store import JSONLExperimentStore
from .symbolic import discover_polynomial_relation, predict_polynomial_relation

__all__ = [
    "AdaptiveMathEngine",
    "ClassicalMLModel",
    "JSONLExperimentStore",
    "discover_patterns",
    "discover_polynomial_relation",
    "predict_adaptive_candidate",
    "predict_polynomial_relation",
    "supported_algorithms",
]
