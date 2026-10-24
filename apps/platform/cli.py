"""Command-line interface for the C-MATH-AI F43 platform."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from wsgiref.simple_server import make_server

from pydantic import ValidationError

from apps.platform.api import create_app
from apps.platform.contracts import PlatformRunRequest
from apps.platform.service import PlatformService
from apps.platform.store import JobConflictError, SQLiteJobStore


def _database_path(value: str | None) -> str:
    return value or os.environ.get(
        "CMATH_PLATFORM_DB", "~/.local/state/c-math-ai/platform.sqlite3"
    )


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))


def _service(path: str | None) -> PlatformService:
    return PlatformService(SQLiteJobStore(_database_path(path)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="c-math-ai",
        description="C-MATH-AI platform: research orchestration and paper-only execution.",
    )
    parser.add_argument("--db", help="SQLite job-store path (or CMATH_PLATFORM_DB)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("health", help="show component readiness, registry counts and job counts")

    algorithms = commands.add_parser("algorithms", help="discover registered algorithm metadata")
    algorithms.add_argument("--category")
    algorithms.add_argument("--lifecycle", choices=("experimental", "implemented", "validated", "deprecated"))
    algorithms.add_argument("--tag")
    algorithms.add_argument("--limit", type=int, default=100)

    run = commands.add_parser("run", help="execute a platform job from a JSON request file")
    run.add_argument("--request", required=True, type=Path)

    job = commands.add_parser("job", help="retrieve a previously persisted job")
    job.add_argument("job_id")

    audit = commands.add_parser("audit", help="verify or print the audit chain for a job")
    audit.add_argument("job_id")
    audit.add_argument("--show", action="store_true", help="include audit events in the output")

    serve = commands.add_parser("serve", help="run the synchronous WSGI API for local/private use")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = _service(args.db)
    try:
        if args.command == "health":
            _print(service.health())
            return 0
        if args.command == "algorithms":
            _print(
                service.algorithms(
                    category=args.category,
                    lifecycle=args.lifecycle,
                    tag=args.tag,
                    limit=args.limit,
                )
            )
            return 0
        if args.command == "run":
            raw = args.request.read_text(encoding="utf-8")
            request = PlatformRunRequest.model_validate_json(raw)
            job = service.submit_job(request)
            _print(job.model_dump(mode="json"))
            return 0 if job.state == "SUCCEEDED" else 1
        if args.command == "job":
            job = service.get_job(args.job_id)
            if job is None:
                print(f"job not found: {args.job_id}", file=sys.stderr)
                return 2
            _print(job.model_dump(mode="json"))
            return 0
        if args.command == "audit":
            verified = service.verify_audit(args.job_id)
            payload: dict[str, Any] = {"job_id": args.job_id, "verified": verified}
            if args.show:
                payload["events"] = service.audit_events(args.job_id)
            _print(payload)
            return 0 if verified else 1
        if args.command == "serve":
            if not 1 <= args.port <= 65_535:
                parser.error("--port must be between 1 and 65535")
            app = create_app(_database_path(args.db))
            with make_server(args.host, args.port, app) as httpd:
                print(
                    f"C-MATH-AI platform API at http://{args.host}:{args.port} "
                    "(paper-only; use a protected reverse proxy outside local development)",
                    flush=True,
                )
                httpd.serve_forever()
            return 0
    except (OSError, ValueError, ValidationError, JobConflictError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
