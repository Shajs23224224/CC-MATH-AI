from datetime import UTC, datetime, timedelta

import pytest

from apps.platform.contracts import (
    MarketBarInput,
    PortfolioAssetMetadata,
    PortfolioPipelineInput,
    PlatformBacktestConfig,
    PlatformRunRequest,
    PlatformScenarioInput,
    PlatformWalkForwardConfig,
    TargetPositionInput,
)
from apps.platform.service import PlatformService
from apps.platform.store import JobConflictError, SQLiteJobStore
from backtest.robustness.contracts import MonteCarloConfig
from core.contracts import DependenceMatrix, Forecast, RiskMetric
from execution.paper import ExecutionQuote, PaperOrderRequest, PaperExecutionPolicy
from portfolio.optimization import PortfolioOptimizationConfig
from portfolio.sizing import PositionSizingConfig
from risk.governor import PairwiseCorrelation, RiskGovernorConfig
from signals.decision import DecisionConfig, DecisionRequest
from signals.ensemble import EnsembleEvidence, EnsembleRequest, combine_evidence


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
        service.submit_job(
            sample_request(job_id=request.job_id).model_copy(update={"symbol": "ABC"})
        )


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


def portfolio_pipeline_input(as_of: datetime) -> PortfolioPipelineInput:
    decision_as_of = as_of - timedelta(seconds=20)
    evidence_time = decision_as_of - timedelta(minutes=1)
    symbols = ("AAA", "BBB")
    expected_returns = (0.04, 0.03)
    decision_requests = []
    for symbol, expected_return in zip(symbols, expected_returns, strict=True):
        ensemble = combine_evidence(
            EnsembleRequest(
                asset_symbol=symbol,
                as_of=decision_as_of,
                horizon="1D",
                data_snapshot_id=f"market-{symbol}-v1",
                evidence=tuple(
                    EnsembleEvidence(
                        source_id=f"{symbol}-model-{index}",
                        source_version="1.0.0",
                        observed_at=evidence_time,
                        horizon="1D",
                        raw_score=score,
                        normalization="bounded",
                        confidence=1.0,
                        reliability=1.0,
                        uncertainty=0.0,
                        correlation_group=f"{symbol}-group-{index}",
                        rationale="Synthetic integration-test evidence.",
                    )
                    for index, score in enumerate((0.8, 0.7), start=1)
                ),
            )
        )
        from core.contracts import Forecast

        decision_requests.append(
            DecisionRequest(
                ensemble=ensemble,
                forecast=Forecast(
                    horizon="1D",
                    expected_return=expected_return,
                    expected_risk=0.1,
                    uncertainty=0.1,
                ),
                forecast_source_id=f"forecast-{symbol}",
                forecast_source_version="1.0.0",
                forecast_observed_at=decision_as_of - timedelta(seconds=30),
                regime="SYNTHETIC",
                model_versions=("model@1.0.0",),
                feature_versions=("feature@1.0.0",),
            )
        )

    calibration = DecisionConfig(
        probability_calibration_status="validated_oos",
        calibration_id="integration-cal-v1",
        calibration_fit_fingerprint="a" * 64,
        calibration_version="1.0.0",
        calibration_data_snapshot_id="fit-snapshot-v1",
        calibration_window_start=datetime(2025, 12, 1, tzinfo=UTC),
        calibration_window_end=datetime(2025, 12, 10, tzinfo=UTC),
        calibration_validation_data_snapshot_id="validation-snapshot-v1",
        calibration_validation_window_start=datetime(2025, 12, 11, tzinfo=UTC),
        calibration_validation_window_end=datetime(2025, 12, 15, tzinfo=UTC),
        calibration_validation_log_loss=0.7,
        calibration_validation_baseline_log_loss=1.0,
    )
    pipeline_assets = (
        PortfolioAssetMetadata(
            asset_symbol="AAA",
            sector="TECH",
            price=100.0,
            volatility=0.2,
            average_daily_volume=100_000.0,
            average_daily_dollar_volume=10_000_000.0,
            lot_size=1.0,
            stop_loss_price=95.0,
            factor_loadings={"MKT": 0.25},
        ),
        PortfolioAssetMetadata(
            asset_symbol="BBB",
            sector="HEALTH",
            price=50.0,
            volatility=0.1,
            average_daily_volume=100_000.0,
            average_daily_dollar_volume=5_000_000.0,
            lot_size=1.0,
            stop_loss_price=47.5,
            factor_loadings={"MKT": 0.25},
        ),
    )
    metrics = tuple(
        RiskMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=(as_of - timedelta(seconds=1)).isoformat(),
            methodology="Synthetic point-in-time integration fixture.",
        )
        for name, value, unit in (
            ("max_drawdown", 0.05, "fraction"),
            ("daily_loss_fraction", 0.01, "fraction"),
            ("max_cvar_fraction", 0.04, "fraction_of_notional"),
            ("data_quality_score", 0.99, "score_0_to_1"),
            ("model_quality_score", 0.95, "score_0_to_1"),
        )
    )
    return PortfolioPipelineInput(
        decisions=tuple(decision_requests),
        decision_config=calibration,
        assets=pipeline_assets,
        covariance=DependenceMatrix(
            assets=symbols,
            method="sample_covariance",
            kind="covariance",
            observations=60,
            start_index=0,
            end_index=60,
            matrix=((0.04, 0.006), (0.006, 0.01)),
        ),
        horizon="1D",
        covariance_as_of=as_of - timedelta(seconds=25),
        covariance_data_snapshot_id="covariance-snapshot-v1",
        optimization_config=PortfolioOptimizationConfig(method="equal_weight"),
        sizing_config=PositionSizingConfig(method="risk_based"),
        equity=100_000.0,
        base_currency="USD",
        sizing_as_of=as_of - timedelta(seconds=2),
        market_data_as_of=as_of - timedelta(seconds=2),
        risk_snapshot_as_of=as_of - timedelta(seconds=1),
        risk_snapshot_id="risk-snapshot-v1",
        model_snapshot_id="model-snapshot-v1",
        risk_metrics=metrics,
        pairwise_correlations=(
            PairwiseCorrelation(asset_a="AAA", asset_b="BBB", correlation=0.35),
        ),
    )


def test_full_f35_to_f38_pipeline_and_paper_order(tmp_path) -> None:
    store = SQLiteJobStore(tmp_path / "portfolio-pipeline.sqlite3")
    service = PlatformService(store)
    request = sample_request(job_id="portfolio-integration-0001")
    as_of = request.as_of
    pipeline = portfolio_pipeline_input(as_of)
    order = PaperOrderRequest(
        asset_symbol="AAA",
        side="BUY",
        quantity=1.0,
        idempotency_key="f43-buy-order-000001",
        strategy_id="f43-integration-test",
        signal_id="f35-decision-aaa",
        submitted_at=as_of,
    )
    quote = ExecutionQuote(
        asset_symbol="AAA",
        bid=99.9,
        ask=100.0,
        timestamp=as_of,
    )
    request = request.model_copy(
        update={
            "portfolio_pipeline": pipeline,
            "risk_policy": RiskGovernorConfig(max_input_age_seconds=300),
            "paper_orders": (order,),
            "paper_quotes": (quote,),
            "paper_initial_cash": 100_000.0,
            "paper_policy": PaperExecutionPolicy(
                commission_bps=1.0,
                market_slippage_bps=2.0,
                max_quote_age_seconds=30,
                max_risk_decision_age_seconds=300,
            ),
        }
    )

    job = service.submit_job(request)
    assert job.state == "SUCCEEDED", job.error_message
    assert job.result is not None
    integration = job.result["portfolio_pipeline"]
    assert len(integration["f35_decisions"]) == 2
    assert integration["f36_optimization"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert integration["f37_sizing"]["total_target_notional"] > 0
    assert integration["f38_risk_decision"]["action"] == "ALLOW"
    assert job.result["paper"]["state"] == "SUCCEEDED"
    assert len(job.result["paper"]["fills"]) == 1
    assert job.result["paper"]["audit_chain_verified"] is True
    assert service.verify_audit(request.job_id) is True
