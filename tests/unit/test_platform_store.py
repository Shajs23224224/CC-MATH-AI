import pytest

from apps.platform.store import JobConflictError, SQLiteJobStore


def test_job_store_idempotency_transitions_and_audit(tmp_path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    first, created = store.create_job("job-store-0001", "a" * 64, '{"x":1}')
    assert created is True
    assert first.state == "PENDING"

    repeated, created_again = store.create_job("job-store-0001", "a" * 64, '{"x":1}')
    assert created_again is False
    assert repeated.job_id == first.job_id

    with pytest.raises(JobConflictError, match="different request fingerprint"):
        store.create_job("job-store-0001", "b" * 64, '{"x":2}')

    store.transition(first.job_id, "RUNNING")
    store.record_event(first.job_id, "TEST_STAGE", {"count": 3, "ok": True})
    succeeded = store.transition(first.job_id, "SUCCEEDED", result={"answer": 42})
    assert succeeded.state == "SUCCEEDED"
    assert succeeded.attempts == 1
    assert succeeded.result == {"answer": 42}
    assert store.verify_audit_chain(first.job_id) is True
    assert len(store.events(first.job_id)) == 4

    with pytest.raises(JobConflictError, match="invalid job transition"):
        store.transition(first.job_id, "RUNNING")


def test_job_store_records_failed_run_and_empty_audit_fails_closed(tmp_path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    record, _ = store.create_job("job-store-0002", "c" * 64, '{"ok":true}')
    assert store.verify_audit_chain(record.job_id) is True
    store.transition(record.job_id, "RUNNING")
    failed = store.transition(
        record.job_id,
        "FAILED",
        error_code="TEST_FAILURE",
        error_message="controlled failure",
    )
    assert failed.state == "FAILED"
    assert failed.error_code == "TEST_FAILURE"
    assert store.verify_audit_chain(record.job_id) is True
    assert store.count_states() == {"PENDING": 0, "RUNNING": 0, "SUCCEEDED": 0, "FAILED": 1}
