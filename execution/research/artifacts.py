"""Shared artifact integrity and atomic persistence helpers for D07."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject arrays/scalars at the root."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with finite-number enforcement and atomic replacement."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic CSV using round-trip-safe float formatting."""
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _safe_artifact_path(directory: Path, filename: object) -> Path:
    """Resolve a manifest filename while refusing traversal and nested paths."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("manifest artifact filename must be a non-empty string")
    candidate = Path(filename)
    if candidate.name != filename or filename in {".", ".."}:
        raise ValueError(f"unsafe artifact filename in manifest: {filename!r}")
    return directory / filename


def verify_manifest_outputs(
    directory: Path,
    manifest_name: str,
    required_files: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify required output hashes and byte counts from an unsigned manifest."""
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"required manifest not found: {manifest_name}")

    manifest = read_json_object(manifest_path)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{manifest_name} outputs must be an object")

    verified: dict[str, str] = {}
    for filename in required_files:
        artifact_path = _safe_artifact_path(directory, filename)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"required artifact not found: {filename}")

        entry = outputs.get(filename)
        if not isinstance(entry, dict):
            raise ValueError(f"{filename} is not declared in {manifest_name}")

        expected_hash = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"{filename} has an invalid SHA-256 entry")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValueError(f"{filename} has an invalid byte-count entry")

        actual_bytes = artifact_path.stat().st_size
        actual_hash = sha256_file(artifact_path)
        if actual_bytes != expected_bytes:
            raise ValueError(f"{filename} byte-count verification failed")
        if actual_hash != expected_hash:
            raise ValueError(f"{filename} SHA-256 verification failed")
        verified[filename] = actual_hash

    return manifest, verified


def manifest_output_entries(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    """Build output hash/size entries for files in a completed run directory."""
    entries: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"cannot manifest missing output: {path}")
        entries[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return entries
