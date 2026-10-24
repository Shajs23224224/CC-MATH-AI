"""Dependency-light WSGI API for the F43 local/private platform service."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from pydantic import ValidationError

from apps.platform.contracts import PlatformRunRequest
from apps.platform.service import PlatformService
from apps.platform.store import JobConflictError, SQLiteJobStore


_MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class PlatformWSGIApplication:
    """Synchronous WSGI endpoints; run behind authentication on trusted networks only."""

    def __init__(self, service: PlatformService) -> None:
        self.service = service

    def __call__(self, environ: dict[str, Any], start_response: Any) -> list[bytes]:
        try:
            method = str(environ.get("REQUEST_METHOD", "GET")).upper()
            path = str(environ.get("PATH_INFO", "/"))
            query = parse_qs(str(environ.get("QUERY_STRING", "")), keep_blank_values=False)

            if method == "GET" and path in {"/health", "/ready"}:
                status, content_type, body = HTTPStatus.OK, "application/json", self.service.health()
            elif method == "GET" and path == "/algorithms":
                limit = int(query.get("limit", ["100"])[0])
                body = self.service.algorithms(
                    category=query.get("category", [None])[0],
                    lifecycle=query.get("lifecycle", [None])[0],
                    tag=query.get("tag", [None])[0],
                    limit=limit,
                )
                status, content_type = HTTPStatus.OK, "application/json"
            elif method == "GET" and path == "/metrics":
                counts = self.service.store.count_states()
                lines = [
                    "# HELP cmath_platform_jobs Number of durable platform jobs by state.",
                    "# TYPE cmath_platform_jobs gauge",
                ]
                lines.extend(
                    f'cmath_platform_jobs{{state="{state.lower()}"}} {count}'
                    for state, count in counts.items()
                )
                status, content_type, body = (
                    HTTPStatus.OK,
                    "text/plain; version=0.0.4; charset=utf-8",
                    "\n".join(lines) + "\n",
                )
            elif method == "POST" and path == "/jobs":
                raw_body = self._read_body(environ)
                request = PlatformRunRequest.model_validate_json(raw_body)
                job = self.service.submit_job(request)
                status, content_type, body = HTTPStatus.OK, "application/json", job.model_dump(mode="json")
            elif method == "GET" and path.startswith("/jobs/"):
                rest = path[len("/jobs/") :]
                if rest.endswith("/audit/verify"):
                    job_id = rest[: -len("/audit/verify")].strip("/")
                    body = {"job_id": job_id, "verified": self.service.verify_audit(job_id)}
                    status, content_type = HTTPStatus.OK, "application/json"
                elif rest.endswith("/audit"):
                    job_id = rest[: -len("/audit")].strip("/")
                    body = {"job_id": job_id, "events": self.service.audit_events(job_id)}
                    status, content_type = HTTPStatus.OK, "application/json"
                else:
                    job_id = rest.strip("/")
                    job = self.service.get_job(job_id)
                    if job is None:
                        raise _HTTPError(HTTPStatus.NOT_FOUND, "job not found")
                    body = job.model_dump(mode="json")
                    status, content_type = HTTPStatus.OK, "application/json"
            else:
                raise _HTTPError(HTTPStatus.NOT_FOUND, "route not found")
        except _HTTPError as exc:
            status, content_type, body = HTTPStatus(exc.status), "application/json", {"error": exc.message}
        except JobConflictError as exc:
            status, content_type, body = HTTPStatus.CONFLICT, "application/json", {"error": str(exc)}
        except ValidationError as exc:
            status, content_type, body = (
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "application/json",
                {"error": "request validation failed", "details": exc.errors(include_url=False, include_context=False)},
            )
        except (ValueError, json.JSONDecodeError) as exc:
            status, content_type, body = HTTPStatus.BAD_REQUEST, "application/json", {"error": str(exc)}
        except Exception:
            # Do not return internal exception details or runtime configuration to callers.
            status, content_type, body = (
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "application/json",
                {"error": "internal platform error"},
            )

        if isinstance(body, bytes):
            encoded = body
        elif isinstance(body, str) and content_type.startswith("text/plain"):
            encoded = body.encode("utf-8")
        else:
            encoded = _json_bytes(body)
        start_response(
            f"{status.value} {status.phrase}",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(encoded))),
                ("Cache-Control", "no-store"),
                ("X-Content-Type-Options", "nosniff"),
            ],
        )
        return [encoded]

    @staticmethod
    def _read_body(environ: dict[str, Any]) -> bytes:
        length_text = str(environ.get("CONTENT_LENGTH", "") or "0")
        try:
            length = int(length_text)
        except ValueError as exc:
            raise ValueError("CONTENT_LENGTH must be an integer") from exc
        if length < 1:
            raise ValueError("request body is required")
        if length > _MAX_REQUEST_BYTES:
            raise _HTTPError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body exceeds 8 MiB")
        stream = environ.get("wsgi.input")
        if stream is None:
            raise ValueError("request body stream is missing")
        body = stream.read(length)
        if len(body) != length:
            raise ValueError("request body was shorter than CONTENT_LENGTH")
        return body


class _HTTPError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def create_app(database_path: str | Path | None = None) -> PlatformWSGIApplication:
    """Create a WSGI app; use CMATH_PLATFORM_DB to select a persistent job store."""
    selected_path = database_path or os.environ.get(
        "CMATH_PLATFORM_DB", "~/.local/state/c-math-ai/platform.sqlite3"
    )
    return PlatformWSGIApplication(PlatformService(SQLiteJobStore(selected_path)))


__all__ = ["PlatformWSGIApplication", "create_app"]
