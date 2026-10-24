import io
import json
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any

from apps.platform.api import create_app
from apps.platform.contracts import MarketBarInput, PlatformRunRequest, TargetPositionInput
from backtest.robustness.contracts import MonteCarloConfig


def _invoke(
    app: Any,
    *,
    method: str = "GET",
    path: str = "/health",
    query: str = "",
    body: bytes = b"",
) -> tuple[str, dict[str, str], bytes]:
    from wsgiref.util import setup_testing_defaults

    environ: dict[str, Any] = {}
    setup_testing_defaults(environ)
    environ.update(
        {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": query,
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }
    )
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = dict(headers)

    response = b"".join(app(environ, start_response))
    return captured["status"], captured["headers"], response


def _request() -> PlatformRunRequest:
    origin = datetime(2026, 2, 1, 14, 30, tzinfo=UTC)
    bars = tuple(
        MarketBarInput(
            timestamp=origin + timedelta(days=i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=10_000.0,
        )
        for i in range(6)
    )
    return PlatformRunRequest(
        job_id="api-smoke-job-0001",
        dataset_id="api-smoke-v1",
        symbol="TEST",
        as_of=bars[-1].timestamp,
        bars=bars,
        targets=(TargetPositionInput(timestamp=bars[0].timestamp, target_units=1),),
        monte_carlo=MonteCarloConfig(simulations=5, seed=1, method="iid", block_size=1),
    )


def test_api_health_and_registry_routes(tmp_path: Path) -> None:
    app = create_app(tmp_path / "api.sqlite3")
    status, headers, body = _invoke(app, path="/health")
    assert status.startswith(str(HTTPStatus.OK.value))
    assert headers["Cache-Control"] == "no-store"
    health = json.loads(body)
    assert health["status"] == "ready"
    assert health["live_execution_enabled"] is False

    status, _, body = _invoke(app, path="/algorithms", query="lifecycle=experimental&limit=2")
    assert status.startswith("200")
    catalog = json.loads(body)
    assert catalog["returned"] <= 2
    assert all(item["lifecycle"] == "experimental" for item in catalog["items"])


def test_api_accepts_idempotent_job_and_exposes_audit(tmp_path: Path) -> None:
    app = create_app(tmp_path / "jobs.sqlite3")
    request = _request()
    status, _, body = _invoke(
        app,
        method="POST",
        path="/jobs",
        body=request.model_dump_json().encode("utf-8"),
    )
    assert status.startswith("200")
    job = json.loads(body)
    assert job["job_id"] == request.job_id
    assert job["state"] == "SUCCEEDED"

    status, _, body = _invoke(app, path=f"/jobs/{request.job_id}/audit/verify")
    assert status.startswith("200")
    assert json.loads(body)["verified"] is True

    status, _, body = _invoke(app, path=f"/jobs/{request.job_id}")
    assert status.startswith("200")
    assert json.loads(body)["result"]["safety"]["live_execution_enabled"] is False


def test_api_validates_request_schema_and_limits_body(tmp_path: Path) -> None:
    app = create_app(tmp_path / "invalid.sqlite3")
    status, _, body = _invoke(app, method="POST", path="/jobs", body=b'{"unknown":true}')
    assert status.startswith(str(HTTPStatus.UNPROCESSABLE_ENTITY.value))
    assert json.loads(body)["error"] == "request validation failed"

    # The limit is checked before reading, so no allocation of an oversized body occurs.
    status, _, body = _invoke(
        app,
        method="POST",
        path="/jobs",
        body=b"x" * (8 * 1024 * 1024 + 1),
    )
    assert status.startswith(str(HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value))
