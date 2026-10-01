"""Feature-store contracts and development implementation."""

from .base import FeatureStore
from .local import LocalFeatureStore

__all__ = [
    "FeatureStore",
    "LocalFeatureStore",
]
