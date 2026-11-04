#!/usr/bin/env python3
"""Start the local CC-MATH-AI training dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from apps.dashboard import (  # noqa: E402
    TrainingEventStore,
    TrainingOrchestrator,
    create_app,
)
from apps.dashboard.training_tasks import register_training_tasks  # noqa: E402
from core.observability.training import TrainingEventRecorder  # noqa: E402


def build_application():
    artifacts = ROOT / "artifacts" / "training"
    artifacts.mkdir(parents=True, exist_ok=True)

    event_store = TrainingEventStore(
        artifacts / "dashboard_events.jsonl"
    )
    recorder = TrainingEventRecorder(
        event_sink=event_store.append
    )

    orchestrator = TrainingOrchestrator(
        recorder,
        max_workers=1,
        max_pending=1,
    )

    register_training_tasks(orchestrator)

    app = create_app(
        recorder,
        event_store=event_store,
        orchestrator=orchestrator,
    )

    return app, orchestrator


def main() -> None:
    app, orchestrator = build_application()

    print("=" * 55, flush=True)
    print("CC-MATH-AI — TRAINING DASHBOARD", flush=True)
    print("URL: http://127.0.0.1:8000/dashboard", flush=True)
    print("API: http://127.0.0.1:8000/docs", flush=True)
    print("Workers: 1", flush=True)
    print("=" * 55, flush=True)

    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
        )
    finally:
        orchestrator.shutdown(
            wait=True,
            cancel_active=True,
        )


if __name__ == "__main__":
    main()
