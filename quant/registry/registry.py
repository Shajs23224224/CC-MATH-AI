"""Thread-safe in-memory algorithm registry with explicit version semantics."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from .contracts import AlgorithmLifecycle, AlgorithmMetadata


class AlgorithmRegistryError(ValueError):
    """Raised when a registry operation would violate registry invariants."""


class AlgorithmNotFoundError(LookupError):
    """Raised when no matching algorithm version is registered."""


class AlgorithmRegistry:
    """Registry metadata only: it never imports or executes algorithm code."""

    def __init__(self, algorithms: Iterable[AlgorithmMetadata] = ()) -> None:
        self._lock = RLock()
        self._entries: dict[tuple[str, str], AlgorithmMetadata] = {}
        self.register_many(algorithms)

    def register(self, metadata: AlgorithmMetadata, *, replace: bool = False) -> None:
        """Register one id/version pair; replacement must be explicitly requested."""
        key = (metadata.algorithm_id, metadata.version)
        with self._lock:
            if key in self._entries and not replace:
                raise AlgorithmRegistryError(
                    f"algorithm {metadata.algorithm_id}@{metadata.version} is already registered"
                )
            self._entries[key] = metadata

    def register_many(self, algorithms: Iterable[AlgorithmMetadata]) -> None:
        """Register a batch atomically; duplicates reject the entire batch."""
        items = tuple(algorithms)
        keys = [(item.algorithm_id, item.version) for item in items]
        if len(keys) != len(set(keys)):
            raise AlgorithmRegistryError("batch contains duplicate algorithm id/version pairs")
        with self._lock:
            conflicts = sorted(key for key in keys if key in self._entries)
            if conflicts:
                algorithm_id, version = conflicts[0]
                raise AlgorithmRegistryError(
                    f"algorithm {algorithm_id}@{version} is already registered"
                )
            self._entries.update(zip(keys, items, strict=True))

    def get(self, algorithm_id: str, version: str | None = None) -> AlgorithmMetadata:
        """Return an exact version, or the highest registered semantic version."""
        with self._lock:
            matches = [
                item for (entry_id, _), item in self._entries.items() if entry_id == algorithm_id
            ]
        if not matches:
            raise AlgorithmNotFoundError(f"algorithm {algorithm_id!r} is not registered")
        if version is not None:
            for item in matches:
                if item.version == version:
                    return item
            raise AlgorithmNotFoundError(f"algorithm {algorithm_id}@{version} is not registered")
        return max(matches, key=lambda item: _version_key(item.version))

    def discover(
        self,
        *,
        category: str | None = None,
        lifecycle: AlgorithmLifecycle | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
        tag: str | None = None,
        include_deprecated: bool = False,
    ) -> tuple[AlgorithmMetadata, ...]:
        """Find algorithms by declared metadata without loading implementations."""
        with self._lock:
            entries = tuple(self._entries.values())
        found = [
            item
            for item in entries
            if (category is None or item.category == category)
            and (lifecycle is None or item.lifecycle == lifecycle)
            and (input_name is None or any(p.name == input_name for p in item.inputs))
            and (output_name is None or any(p.name == output_name for p in item.outputs))
            and (tag is None or tag in item.tags)
            and (include_deprecated or item.lifecycle != AlgorithmLifecycle.DEPRECATED)
        ]
        return tuple(
            sorted(found, key=lambda item: (item.algorithm_id, _version_key(item.version)))
        )

    def validate_compatibility(
        self, source: AlgorithmMetadata, target: AlgorithmMetadata
    ) -> tuple[str, ...]:
        """Return compatibility errors for matching source-output/target-input names."""
        errors: list[str] = []
        source_outputs = {port.name: port for port in source.outputs}
        for required in target.inputs:
            actual = source_outputs.get(required.name)
            if actual is None:
                if required.required:
                    errors.append(f"missing required input {required.name!r}")
                continue
            if actual.dtype != required.dtype:
                errors.append(
                    f"input {required.name!r} expects {required.dtype}, got {actual.dtype}"
                )
            if required.unit is not None and actual.unit != required.unit:
                errors.append(
                    f"input {required.name!r} expects unit {required.unit!r}, got {actual.unit!r}"
                )
        return tuple(errors)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def snapshot(self) -> tuple[AlgorithmMetadata, ...]:
        """Return an immutable, deterministically ordered metadata snapshot."""
        with self._lock:
            return tuple(
                sorted(
                    self._entries.values(),
                    key=lambda item: (item.algorithm_id, _version_key(item.version)),
                )
            )


def _version_key(version: str) -> tuple[int, int, int, int, str]:
    """Sort semantic versions numerically; prereleases precede their release."""
    core, separator, suffix = version.partition("-")
    major, minor, patch = (int(part) for part in core.split("."))
    return major, minor, patch, 0 if separator else 1, suffix
