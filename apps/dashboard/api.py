"""Read-only HTTP API and resumable SSE transport for the Training Dashboard (TD03)."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from apps.dashboard.event_store import TrainingEventStore
from apps.dashboard.service import TrainingDashboardService
from core.contracts.training import TrainingEvent, TrainingRunSnapshot
from core.observability.training import (
    TrainingEventRecorder,
    TrainingRunNotFoundError,
)

_DEFAULT_HEARTBEAT_SECONDS = 15.0
_MAX_EVENT_PAGE_SIZE = 1_000


def _encode_sse(event: TrainingEvent) -> str:
    """Serialize one event using its per-run sequence as the reconnect cursor."""
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type.lower()}\n"
        f"data: {event.model_dump_json()}\n\n"
    )


def create_app(
    recorder: TrainingEventRecorder,
    *,
    event_store: TrainingEventStore | None = None,
    heartbeat_interval_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
) -> FastAPI:
    """Build an HTTP application around explicitly supplied observability services.

    Inject the recorder shared by the training process. When an event store is
    supplied, historical run snapshots and events remain queryable after restart.
    Run lifecycle mutation stays in the training/application layer; this API is
    intentionally read-only.
    """
    if not math.isfinite(heartbeat_interval_seconds) or heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")

    service = TrainingDashboardService(recorder, event_store=event_store)
    app = FastAPI(
        title="C-MATH-AI Training Dashboard API",
        version="0.4.0",
        description="Read-only training-run observability with resumable SSE events.",
    )
    app.state.training_recorder = recorder
    app.state.training_dashboard_service = service

    @app.get("/healthz", tags=["system"])
    def health() -> dict[str, str]:
        """Return process liveness without implying model or storage health."""
        return {"status": "ok"}

    @app.get(
        "/api/v1/runs",
        response_model=list[TrainingRunSnapshot],
        tags=["training"],
    )
    def list_runs() -> tuple[TrainingRunSnapshot, ...]:
        """List known in-memory and persisted runs, most recently updated first."""
        return service.list_runs()

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=TrainingRunSnapshot,
        tags=["training"],
    )
    def get_run(run_id: str) -> TrainingRunSnapshot:
        """Return the latest snapshot for one run."""
        try:
            return service.snapshot(run_id)
        except TrainingRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Training run not found") from exc

    @app.get(
        "/api/v1/runs/{run_id}/events",
        response_model=list[TrainingEvent],
        tags=["training"],
    )
    def list_run_events(
        run_id: str,
        after_sequence: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=_MAX_EVENT_PAGE_SIZE)] = 200,
    ) -> tuple[TrainingEvent, ...]:
        """Return events after an exclusive per-run sequence cursor."""
        try:
            return service.events(
                run_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        except TrainingRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Training run not found") from exc

    @app.get(
        "/api/v1/runs/{run_id}/events/stream",
        tags=["training"],
        response_class=StreamingResponse,
    )
    async def stream_run_events(
        request: Request,
        run_id: str,
        after_sequence: Annotated[int | None, Query(ge=0)] = None,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        """Stream ordered run events and resume from the last delivered sequence."""
        try:
            service.snapshot(run_id)
        except TrainingRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Training run not found") from exc

        cursor = after_sequence or 0
        if last_event_id is not None:
            try:
                cursor = int(last_event_id)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID must be a non-negative sequence number",
                ) from exc
            if cursor < 0:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID must be a non-negative sequence number",
                )

        live_run = service.is_live_run(run_id)

        async def event_stream() -> AsyncIterator[str]:
            nonlocal cursor
            yield "retry: 3000\n\n"

            if not live_run:
                # Recovered runs are immutable history in this process: replay then close.
                for event in service.events(run_id, after_sequence=cursor):
                    cursor = event.sequence
                    yield _encode_sse(event)
                yield "event: stream_end\\ndata: {}\\n\\n"
                return

            loop = asyncio.get_running_loop()
            notification = asyncio.Event()

            def wake_listener(event: TrainingEvent) -> None:
                if event.run_id == run_id:
                    try:
                        loop.call_soon_threadsafe(notification.set)
                    except RuntimeError:
                        # The client disconnected while a producer was notifying us.
                        return

            unsubscribe = service.subscribe(wake_listener)
            try:
                while not await request.is_disconnected():
                    # Clear before reading: an event arriving during the read either
                    # appears in the read or sets the notification for the next wait.
                    notification.clear()
                    pending = service.events(run_id, after_sequence=cursor)

                    for event in pending:
                        cursor = event.sequence
                        yield _encode_sse(event)

                    if pending:
                        continue

                    try:
                        await asyncio.wait_for(
                            notification.wait(),
                            timeout=heartbeat_interval_seconds,
                        )
                    except TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                unsubscribe()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/", include_in_schema=False)
    def dashboard_redirect() -> RedirectResponse:
        """Send the service root to the local dashboard interface."""
        return RedirectResponse(url="/dashboard/", status_code=307)

    static_directory = Path(__file__).with_name("static")
    app.mount(
        "/dashboard",
        StaticFiles(directory=static_directory, html=True),
        name="training-dashboard",
    )

    return app


__all__ = ["create_app"]
