"""SQLite-backed F43 job state and tamper-evident audit event store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from apps.platform.contracts import PlatformJobRecord

JobTransition = Literal["RUNNING", "SUCCEEDED", "FAILED"]
_ZERO_HASH = "0" * 64


class JobConflictError(ValueError):
    """Raised when a job identifier is reused for different input or an invalid transition."""


def canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically and reject NaN/Infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


class SQLiteJobStore:
    """Transactional job metadata and per-job SHA-256 audit chain.

    The store persists inputs, summaries and state transitions in SQLite. It never stores
    model weights, provider credentials or market-data files. Use a managed database and
    access controls before exposing an application beyond a trusted network.
    """

    def __init__(self, database_path: str | Path = "var/cmath-platform.sqlite3") -> None:
        raw_path = str(database_path)
        if raw_path != ":memory:":
            path = Path(raw_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(path)
        self.database_path = raw_path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(raw_path, check_same_thread=False, timeout=10.0)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            if raw_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS platform_jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL CHECK(state IN ('PENDING','RUNNING','SUCCEEDED','FAILED')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_platform_jobs_updated
                    ON platform_jobs(updated_at DESC);
                CREATE TABLE IF NOT EXISTS platform_audit_events (
                    job_id TEXT NOT NULL REFERENCES platform_jobs(job_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    PRIMARY KEY(job_id, sequence)
                );
                """
            )
            self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    def create_job(
        self,
        job_id: str,
        fingerprint: str,
        request_json: str,
    ) -> tuple[PlatformJobRecord, bool]:
        """Create a PENDING job, or return an idempotent existing record."""
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    "SELECT * FROM platform_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is not None:
                    if row["request_fingerprint"] != fingerprint:
                        raise JobConflictError(
                            "job_id already exists with a different request fingerprint"
                        )
                    self._connection.commit()
                    return self._record(row), False

                now = utc_now().isoformat()
                cursor.execute(
                    """
                    INSERT INTO platform_jobs (
                        job_id, state, created_at, updated_at, request_fingerprint,
                        request_json, attempts, version
                    ) VALUES (?, 'PENDING', ?, ?, ?, ?, 0, 1)
                    """,
                    (job_id, now, now, fingerprint, request_json),
                )
                self._append_event(
                    cursor,
                    job_id,
                    "JOB_CREATED",
                    now,
                    {"request_fingerprint": fingerprint},
                )
                self._connection.commit()
                result = cursor.execute(
                    "SELECT * FROM platform_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                assert result is not None
                return self._record(result), True
            except Exception:
                self._connection.rollback()
                raise

    def transition(
        self,
        job_id: str,
        state: JobTransition,
        *,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> PlatformJobRecord:
        """Advance a job once; state changes and their audit events commit atomically."""
        allowed = {
            ("PENDING", "RUNNING"),
            ("RUNNING", "SUCCEEDED"),
            ("RUNNING", "FAILED"),
        }
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    "SELECT * FROM platform_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown platform job: {job_id}")
                previous_state = str(row["state"])
                if (previous_state, state) not in allowed:
                    raise JobConflictError(
                        f"invalid job transition: {previous_state} -> {state}"
                    )
                if state == "SUCCEEDED" and result is None:
                    raise ValueError("SUCCEEDED jobs require a result payload")
                if state == "FAILED" and not error_code:
                    raise ValueError("FAILED jobs require an error_code")
                now = utc_now().isoformat()
                result_json = None if result is None else canonical_json(result)
                attempts = int(row["attempts"]) + (1 if state == "RUNNING" else 0)
                cursor.execute(
                    """
                    UPDATE platform_jobs
                    SET state = ?, updated_at = ?, attempts = ?, version = version + 1,
                        result_json = ?, error_code = ?, error_message = ?
                    WHERE job_id = ? AND state = ?
                    """,
                    (
                        state,
                        now,
                        attempts,
                        result_json,
                        error_code,
                        error_message,
                        job_id,
                        previous_state,
                    ),
                )
                if cursor.rowcount != 1:
                    raise JobConflictError("job state changed concurrently")
                self._append_event(
                    cursor,
                    job_id,
                    f"JOB_{state}",
                    now,
                    {
                        "from": previous_state,
                        "to": state,
                        "error_code": error_code,
                        "error_message": error_message,
                        "result_fingerprint": (
                            _digest(result_json) if result_json is not None else None
                        ),
                    },
                )
                self._connection.commit()
                updated = cursor.execute(
                    "SELECT * FROM platform_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                assert updated is not None
                return self._record(updated)
            except Exception:
                self._connection.rollback()
                raise

    def record_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one audit entry to the job-specific tamper-evident chain."""
        if not event_type or len(event_type) > 100:
            raise ValueError("event_type must contain 1–100 characters")
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                if cursor.execute(
                    "SELECT 1 FROM platform_jobs WHERE job_id = ?", (job_id,)
                ).fetchone() is None:
                    raise KeyError(f"unknown platform job: {job_id}")
                self._append_event(cursor, job_id, event_type, utc_now().isoformat(), payload)
                self._connection.commit()
                row = cursor.execute(
                    """
                    SELECT * FROM platform_audit_events
                    WHERE job_id = ? ORDER BY sequence DESC LIMIT 1
                    """,
                    (job_id,),
                ).fetchone()
                assert row is not None
                return self._event_record(row)
            except Exception:
                self._connection.rollback()
                raise

    def get_job(self, job_id: str) -> PlatformJobRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM platform_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return None if row is None else self._record(row)

    def list_jobs(self, limit: int = 100) -> tuple[PlatformJobRecord, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM platform_jobs ORDER BY created_at DESC, job_id LIMIT ?",
                (limit,),
            ).fetchall()
            return tuple(self._record(row) for row in rows)

    def count_states(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT state, COUNT(*) AS count FROM platform_jobs GROUP BY state"
            ).fetchall()
            counts = {str(row["state"]): int(row["count"]) for row in rows}
        return {
            state: counts.get(state, 0)
            for state in ("PENDING", "RUNNING", "SUCCEEDED", "FAILED")
        }

    def events(self, job_id: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            if self._connection.execute(
                "SELECT 1 FROM platform_jobs WHERE job_id = ?", (job_id,)
            ).fetchone() is None:
                raise KeyError(f"unknown platform job: {job_id}")
            rows = self._connection.execute(
                """
                SELECT * FROM platform_audit_events
                WHERE job_id = ? ORDER BY sequence
                """,
                (job_id,),
            ).fetchall()
            return tuple(self._event_record(row) for row in rows)

    def verify_audit_chain(self, job_id: str) -> bool:
        """Check sequence, previous-link and canonical content hashes for one job."""
        try:
            events = self.events(job_id)
        except KeyError:
            return False
        previous_hash = _ZERO_HASH
        for expected_sequence, event in enumerate(events, start=1):
            if event["sequence"] != expected_sequence or event["previous_hash"] != previous_hash:
                return False
            material = canonical_json(
                {
                    "job_id": job_id,
                    "sequence": expected_sequence,
                    "event_type": event["event_type"],
                    "timestamp": event["timestamp"],
                    "payload_json": event["payload_json"],
                    "previous_hash": previous_hash,
                }
            )
            if _digest(material) != event["event_hash"]:
                return False
            previous_hash = event["event_hash"]
        return bool(events)

    @staticmethod
    def _append_event(
        cursor: sqlite3.Cursor,
        job_id: str,
        event_type: str,
        timestamp: str,
        payload: dict[str, Any],
    ) -> None:
        last = cursor.execute(
            """
            SELECT sequence, event_hash FROM platform_audit_events
            WHERE job_id = ? ORDER BY sequence DESC LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        sequence = 1 if last is None else int(last["sequence"]) + 1
        previous_hash = _ZERO_HASH if last is None else str(last["event_hash"])
        payload_json = canonical_json(payload)
        material = canonical_json(
            {
                "job_id": job_id,
                "sequence": sequence,
                "event_type": event_type,
                "timestamp": timestamp,
                "payload_json": payload_json,
                "previous_hash": previous_hash,
            }
        )
        cursor.execute(
            """
            INSERT INTO platform_audit_events
            (job_id, sequence, event_type, timestamp, payload_json, previous_hash, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                sequence,
                event_type,
                timestamp,
                payload_json,
                previous_hash,
                _digest(material),
            ),
        )

    @staticmethod
    def _event_record(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(str(row["payload_json"]))
        return {
            "job_id": str(row["job_id"]),
            "sequence": int(row["sequence"]),
            "event_type": str(row["event_type"]),
            "timestamp": str(row["timestamp"]),
            "payload_json": str(row["payload_json"]),
            "payload": payload,
            "previous_hash": str(row["previous_hash"]),
            "event_hash": str(row["event_hash"]),
        }

    @staticmethod
    def _record(row: sqlite3.Row) -> PlatformJobRecord:
        result_json = row["result_json"]
        return PlatformJobRecord(
            job_id=str(row["job_id"]),
            state=str(row["state"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            request_fingerprint=str(row["request_fingerprint"]),
            attempts=int(row["attempts"]),
            version=int(row["version"]),
            result=None if result_json is None else json.loads(str(result_json)),
            error_code=None if row["error_code"] is None else str(row["error_code"]),
            error_message=None if row["error_message"] is None else str(row["error_message"]),
        )


__all__ = ["JobConflictError", "SQLiteJobStore", "canonical_json"]
