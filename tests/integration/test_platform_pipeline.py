from datetime import UTC, datetime, timedelta

import pytest

from apps.platform.contracts import (
    MarketBarInput,
    PlatformBacktestConfig,
    PlatformRunRequest,
    PlatformScenarioInput,
    PlatformWalkForwardConfig,
    TargetPositionInput,
)
from apps.platform.service import PlatformService
from apps.platform.store import JobConflictError, SQLiteJobStore
from backtest.robustness.contracts import MonteCarloConfig


def sample_request(*, job_id: str = "platform-smoke-0001") -> PlatformRunRequest:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    closes = (100.0, 101.0, 100.5, 102.0, 103.0, 101.5, 104.0, 105.0, 103.5, 106.0)
    bars = []
    for index, close in enumerate(closes):
        open_price = close - 0.25
        bars.append(
            MarketBarInput(
                timestamp=start + timedelta(days=index),
                open=open_price,
                high=max(open_price, close) + 0.5,
                low=min(open_price, close) - 0.5,
                close=close,
                volume=1_000_000.0 + index,
            )
        )
    targets = (
        TargetPositionInput(timestamp=bars[0].timestamp, target_units=1),
        TargetPositionInput(timestamp=bars[5].timestamp, target_units=0),
    )
    return PlatformRunRequest(
        job_id=job_id,
        dataset_id="synthetic-smoke-data-v1",
        symbol="xyz",
        as_of=bars[-1].timestamp,
        bars=tuple(bars),
        targets=targets,
        backtest=PlatformBacktestConfig(initial_capital=10_000.0),
        monte_carlo=MonteCarloConfig(simulations=20, seed=7, method="iid", block_size=2),
        adversarial_scenarios=(
            PlatformScenarioInput(
                name="higher-friction",
                extra_commission_bps=5.0,
                extra_slippage_bps=10.0,
            ),
        ),
        walk_forward=PlatformWalkForwardConfig(enabled=True),
    )


def test_platform_pipeline_is_reproducible_and_persisted(tmp_path) -> None:
    store = SQLiteJobStore(tmp_path / "platform.sqlite3")
    service = PlatformService(store)
    request = sample_request()

    first = service.submit_job(request)
    assert first.state == "SUCCEEDED", first.error_message
    assert first.attempts == 1
    assert first.result is not None
    assert first.result["safety"]["live_execution_enabled"] is False
    assert first.result["dataset"]["quality"]["is_usable"] is True
    assert first.result["dataset"]["features"]["generated_feature_ids"] == [
        "close_return",
        "return_volatility_5",
    ]
    assert first.result["backtest"]["equity_points"] == len(request.bars)
    assert first.result["walk_forward"]["state"] == "SUCCEEDED"
    assert first.result["robustness"]["scenario_count"] == 1
    assert first.result["robustness"]["fingerprint"]
    assert service.verify_audit(request.job_id) is True

    audit_count = len(service.audit_events(request.job_id))
    repeated = service.submit_job(request)
    assert repeated.state == "SUCCEEDED"
    assert repeated.request_fingerprint == first.request_fingerprint
    assert len(service.audit_events(request.job_id)) == audit_count

    with pytest.raises(JobConflictError, match="different request fingerprint"):
        service.submit_job(sample_request(job_id=request.job_id).model_copy(update={"symbol": "ABC"}))


def test_platform_catalog_distinguishes_experimental_metadata(tmp_path) -> None:
    service = PlatformService(SQLiteJobStore(tmp_path / "catalog.sqlite3"))
    health = service.health()
    assert health["status"] == "ready"
    assert health["live_execution_enabled"] is False
    catalog = health["registry"]
    assert catalog["total_entries"] > 0
    assert "experimental" in catalog["lifecycle_counts"]
    items = service.algorithms(lifecycle="experimental", limit=5)
    assert items["returned"] <= 5
    assert all(item["lifecycle"] == "experimental" for item in items["items"])


def test_quality_gate_failure_is_recorded_without_backtest(tmp_path) -> None:
    store = SQLiteJobStore(tmp_path / "gate.sqlite3")
    service = PlatformService(store)
    req = sample_request(job_id="quality-gate-0001")
    # Request schema permits valid individual bars; inject a deliberately corrupt source
    # by replacing the data-quality engine with an explicit fail-closed test double.
    class RejectQuality:
        def check_frame(self, *args, **kwargs):
            from core.contracts import DataQualityReport, DataQualityStatus

            return DataQualityReport(
                dataset_id=kwargs["dataset_id"],
                checked_at=datetime.now(UTC),
                rows=len(args[0]),
                accepted_rows=0,
                rejected_rows=len(args[0]),
                status=DataQualityStatus.INVALID,
                issues=(),
                leakage_detected=False,
                survivorship_bias_risk=False,
            )

    service.quality_engine = RejectQuality()  # type: ignore[assignment]
    failed = service.submit_job(req)
    assert failed.state == "FAILED"
    assert failed.error_code == "QUALITY_GATE_REJECTED"
    assert failed.result is None
    assert service.verify_audit(req.job_id) is True
