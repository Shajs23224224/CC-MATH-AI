"""F43 vertical-slice orchestration over existing quantitative and risk components."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

import pandas as pd

from apps.platform.contracts import (
    PlatformJobRecord,
    PlatformRunRequest,
    PlatformStageResult,
)
from apps.platform.store import JobConflictError, SQLiteJobStore, canonical_json

from backtest.engine import BacktestConfig, Bar, run_backtest
from backtest.robustness import ScenarioSpec, run_robustness
from backtest.walk_forward import run_walk_forward
from backtest.walk_forward.splits import WalkForwardConfig
from core.contracts import FeatureDefinition
from core.contracts.enums import DataQualityStatus
from core.contracts.feature_engineering import FeatureEngineeringConfig, FeatureTransformSpec
from data.quality import DataQualityEngine
from execution.paper import PaperTradingEngine
from ml.features.engineering import FeatureEngineer
from quant.registry import AlgorithmLifecycle, builtin_algorithms, discover_algorithms
from risk.governor import evaluate_risk_governor

_LOGGER = logging.getLogger("c_math_ai.platform")


class PlatformExecutionError(RuntimeError):
    """Expected failure at a named platform stage."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _metrics(backtest: Any) -> dict[str, Any]:
    """Return stable, compact F39 metrics without serializing the entire equity curve."""
    return {
        "initial_capital": backtest.initial_capital,
        "final_equity": backtest.final_equity,
        "total_return": backtest.total_return,
        "max_drawdown": backtest.max_drawdown,
        "sharpe_ratio": backtest.sharpe_ratio,
        "sortino_ratio": backtest.sortino_ratio,
        "turnover": backtest.turnover,
        "trades": backtest.trades,
        "commissions": backtest.commissions,
        "slippage_costs": backtest.slippage_costs,
        "equity_points": len(backtest.equity_curve),
        "fills": len(backtest.fills),
    }


def _catalog_summary() -> dict[str, Any]:
    entries = builtin_algorithms()
    counts = {
        lifecycle.value: sum(item.lifecycle == lifecycle for item in entries)
        for lifecycle in AlgorithmLifecycle
    }
    manifest = [
        item.model_dump(mode="json")
        for item in sorted(entries, key=lambda item: (item.algorithm_id, item.version))
    ]
    return {
        "total_entries": len(entries),
        "lifecycle_counts": counts,
        "implemented_or_validated": counts.get("implemented", 0) + counts.get("validated", 0),
        "catalog_fingerprint": _fingerprint(manifest),
        "catalog_semantics": (
            "Metadata entries only; lifecycle=experimental is not an implemented algorithm, "
            "and registry presence is not evidence of investment performance."
        ),
    }


class PlatformService:
    """Idempotent F43 pipeline service with SQLite job state and audit lineage.

    This first release composes validated market data, feature engineering, registry
    inventory, F39 backtesting, an optional F40 persistence baseline, F41 robustness,
    F38 risk evaluation and F42 paper trading. Strategy target positions remain explicit
    inputs from the caller; this service does not invent decisions or run live execution.
    """

    def __init__(self, store: SQLiteJobStore | None = None) -> None:
        self.store = store or SQLiteJobStore()
        self.quality_engine = DataQualityEngine()
        self.feature_engineer = FeatureEngineer(
            FeatureEngineeringConfig(timestamp_column="timestamp", asset_column="symbol")
        )

    def health(self) -> dict[str, Any]:
        """Return readiness facts without revealing configuration secrets."""
        catalog = _catalog_summary()
        jobs = self.store.count_states()
        return {
            "service": "c-math-ai",
            "release": "1.0.0-rc1",
            "status": "ready",
            "execution_mode": "paper-only",
            "live_execution_enabled": False,
            "registry": catalog,
            "jobs": jobs,
        }

    def algorithms(
        self,
        *,
        category: str | None = None,
        lifecycle: str | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Discover catalog metadata with explicit lifecycle semantics."""
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        selected_lifecycle = None
        if lifecycle is not None:
            try:
                selected_lifecycle = AlgorithmLifecycle(lifecycle)
            except ValueError as exc:
                raise ValueError(
                    "lifecycle must be experimental, implemented, validated or deprecated"
                ) from exc
        entries = discover_algorithms(
            category=category,
            lifecycle=selected_lifecycle,
            tag=tag,
            include_deprecated=selected_lifecycle == AlgorithmLifecycle.DEPRECATED,
        )
        results = [
            item.model_dump(mode="json")
            for item in entries[:limit]
        ]
        return {
            "count": len(entries),
            "returned": len(results),
            "limit": limit,
            "items": results,
            "catalog_fingerprint": _catalog_summary()["catalog_fingerprint"],
        }

    def submit_job(self, request: PlatformRunRequest) -> PlatformJobRecord:
        """Execute or return an idempotent job using a stable input fingerprint."""
        serialized_request = request.model_dump(mode="json")
        request_json = canonical_json(serialized_request)
        fingerprint = _fingerprint(serialized_request)
        record, created = self.store.create_job(request.job_id, fingerprint, request_json)
        if not created:
            if record.state in {"SUCCEEDED", "FAILED"}:
                return record
            if record.state == "RUNNING":
                raise JobConflictError("job is already running in another worker")

        self.store.transition(request.job_id, "RUNNING")
        self.store.record_event(
            request.job_id,
            "PIPELINE_STARTED",
            {
                "dataset_id": request.dataset_id,
                "symbol": request.symbol,
                "bar_count": len(request.bars),
                "request_fingerprint": fingerprint,
            },
        )
        try:
            result = self._execute(request, fingerprint)
            self.store.transition(request.job_id, "SUCCEEDED", result=result)
            return self.store.get_job(request.job_id)  # type: ignore[return-value]
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, PlatformExecutionError)
                else type(exc).__name__.upper()
            )
            message = str(exc) or type(exc).__name__
            _LOGGER.exception("platform job %s failed at runtime", request.job_id)
            self.store.record_event(
                request.job_id,
                "PIPELINE_FAILED",
                {"error_code": code, "error_message": message},
            )
            self.store.transition(
                request.job_id,
                "FAILED",
                error_code=code[:160],
                error_message=message[:2_000],
            )
            return self.store.get_job(request.job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> PlatformJobRecord | None:
        return self.store.get_job(job_id)

    def audit_events(self, job_id: str) -> tuple[dict[str, Any], ...]:
        return self.store.events(job_id)

    def verify_audit(self, job_id: str) -> bool:
        return self.store.verify_audit_chain(job_id)

    def _stage(self, job_id: str, name: str, summary: dict[str, Any]) -> PlatformStageResult:
        self.store.record_event(job_id, f"STAGE_{name.upper()}_SUCCEEDED", summary)
        return PlatformStageResult(stage=name, state="SUCCEEDED", summary=summary)

    def _execute(self, request: PlatformRunRequest, fingerprint: str) -> dict[str, Any]:
        bars: tuple[Bar, ...] = tuple(item.to_bar() for item in request.bars)
        frame = pd.DataFrame(
            [
                {
                    "timestamp": item.timestamp.isoformat(),
                    "symbol": request.symbol,
                    "open": item.open,
                    "high": item.high,
                    "low": item.low,
                    "close": item.close,
                    "volume": item.volume,
                }
                for item in request.bars
            ]
        )
        stages: list[PlatformStageResult] = []

        self.store.record_event(
            request.job_id,
            "MARKET_DATA_ACCEPTED",
            {
                "dataset_id": request.dataset_id,
                "input_rows": len(frame),
                "market_fingerprint": _fingerprint(frame.to_dict(orient="records")),
            },
        )
        quality = self.quality_engine.check_frame(
            frame,
            dataset_id=request.dataset_id,
            required_columns=("timestamp", "symbol", "open", "high", "low", "close", "volume"),
            key_columns=("symbol", "timestamp"),
            numeric_columns=("open", "high", "low", "close", "volume"),
            timestamp_column="timestamp",
            as_of=request.as_of,
            expected_symbols=(request.symbol,),
            symbol_column="symbol",
            survivorship_metadata_present=False,
        )
        quality_summary = {
            "status": quality.status.value,
            "accepted_rows": quality.accepted_rows,
            "rejected_rows": quality.rejected_rows,
            "issue_codes": tuple(item.code for item in quality.issues),
            "leakage_detected": quality.leakage_detected,
            "survivorship_bias_risk": quality.survivorship_bias_risk,
            "is_usable": quality.is_usable,
        }
        if quality.status == DataQualityStatus.INVALID or not quality.is_usable:
            self.store.record_event(request.job_id, "QUALITY_GATE_REJECTED", quality_summary)
            raise PlatformExecutionError(
                "QUALITY_GATE_REJECTED",
                "market data failed the data-quality gate; inspect the audit event for issue codes",
            )
        stages.append(self._stage(request.job_id, "quality", quality_summary))

        specifications = (
            FeatureTransformSpec(
                definition=FeatureDefinition(
                    feature_id="close_return",
                    name="Close-to-close return",
                    version="1.0.0",
                    description="One-period close-to-close simple return.",
                    dtype="float64",
                    source_columns=("close",),
                    frequency="bar",
                    lookback=1,
                    point_in_time=True,
                    methodology="close[t] / close[t-1] - 1",
                ),
                operation="pct_change",
                category="price_return",
                periods=1,
            ),
            FeatureTransformSpec(
                definition=FeatureDefinition(
                    feature_id="return_volatility_5",
                    name="Five-bar return volatility",
                    version="1.0.0",
                    description="Trailing population standard deviation of close returns.",
                    dtype="float64",
                    source_columns=("close_return",),
                    frequency="bar",
                    lookback=5,
                    point_in_time=True,
                    methodology="rolling standard deviation of close_return over five bars",
                ),
                operation="rolling_std",
                category="volatility",
                window=5,
            ),
        )
        featured, feature_report = self.feature_engineer.transform(
            frame, specifications, dataset_id=request.dataset_id
        )
        stages.append(
            self._stage(
                request.job_id,
                "features",
                {
                    "pipeline_id": feature_report.pipeline_id,
                    "generated_feature_ids": feature_report.generated_feature_ids,
                    "input_fingerprint": feature_report.input_fingerprint,
                    "output_fingerprint": feature_report.output_fingerprint,
                    "point_in_time_safe": feature_report.point_in_time_safe,
                    "warmup_missing_rows": feature_report.rows_with_warmup_missing_values,
                },
            )
        )

        catalog = _catalog_summary()
        stages.append(
            self._stage(
                request.job_id,
                "algorithm_registry",
                {
                    "total_entries": catalog["total_entries"],
                    "lifecycle_counts": catalog["lifecycle_counts"],
                    "catalog_fingerprint": catalog["catalog_fingerprint"],
                },
            )
        )

        targets = {item.timestamp: item.target_units for item in request.targets}
        backtest_config = BacktestConfig(
            initial_capital=request.backtest.initial_capital,
            commission_bps=request.backtest.commission_bps,
            slippage_bps=request.backtest.slippage_bps,
            periods_per_year=request.backtest.periods_per_year,
        )
        backtest = run_backtest(bars, targets, backtest_config)
        backtest_summary = _metrics(backtest)
        stages.append(self._stage(request.job_id, "backtest", backtest_summary))

        walk_forward_result: dict[str, Any]
        if not request.walk_forward.enabled:
            walk_forward_result = {"state": "SKIPPED", "reason": "disabled by request"}
            stages.append(
                PlatformStageResult(
                    stage="walk_forward",
                    state="SKIPPED",
                    summary=walk_forward_result,
                )
            )
        elif len(bars) < 4:
            walk_forward_result = {
                "state": "SKIPPED",
                "reason": "at least four bars are required for an OOS split",
            }
            stages.append(
                PlatformStageResult(stage="walk_forward", state="SKIPPED", summary=walk_forward_result)
            )
        else:
            train_size = request.walk_forward.initial_train_size
            test_size = request.walk_forward.test_size
            if train_size is None or test_size is None:
                train_size = max(2, len(bars) // 2)
                test_size = len(bars) - train_size - request.walk_forward.gap
            if test_size < 2 or train_size + request.walk_forward.gap + test_size > len(bars):
                walk_forward_result = {
                    "state": "SKIPPED",
                    "reason": "not enough bars for the requested train/gap/test split",
                }
                stages.append(
                    PlatformStageResult(
                        stage="walk_forward", state="SKIPPED", summary=walk_forward_result
                    )
                )
            else:
                wf_config = WalkForwardConfig(
                    initial_train_size=train_size,
                    test_size=test_size,
                    mode=request.walk_forward.mode,
                    gap=request.walk_forward.gap,
                    anchor_index=request.walk_forward.anchor_index,
                )

                def fit(train: tuple[Bar, ...]) -> int:
                    # This persistence baseline consults only the last training timestamp.
                    return targets.get(train[-1].timestamp, 0)

                def predict(model: int, history: Any) -> int:
                    # The same trained target is held through the OOS window; no future label read.
                    return model

                wf = run_walk_forward(
                    bars,
                    wf_config,
                    fit=fit,
                    predict=predict,
                    backtest_config=backtest_config,
                    strategy_version=request.walk_forward.strategy_version,
                )
                walk_forward_result = {
                    "state": "SUCCEEDED",
                    "strategy_version": wf.strategy_version,
                    "fingerprint": wf.fingerprint,
                    "window_count": wf.summary.window_count,
                    "summary": asdict(wf.summary),
                    "windows": [
                        {
                            "number": item.number,
                            "train_start": item.train_start.isoformat(),
                            "train_end": item.train_end.isoformat(),
                            "test_start": item.test_start.isoformat(),
                            "test_end": item.test_end.isoformat(),
                            "metrics": _metrics(item.backtest),
                        }
                        for item in wf.windows
                    ],
                    "baseline_semantics": (
                        "OOS target-persistence baseline, not the supplied strategy's historical "
                        "out-of-sample performance."
                    ),
                }
                stages.append(self._stage(request.job_id, "walk_forward", {
                    "fingerprint": wf.fingerprint,
                    "window_count": wf.summary.window_count,
                    "strategy_version": wf.strategy_version,
                }))

        scenarios = tuple(
            ScenarioSpec(**item.model_dump()) for item in request.adversarial_scenarios
        )
        robustness = run_robustness(
            bars,
            targets,
            backtest_config=backtest_config,
            monte_carlo_config=request.monte_carlo,
            scenarios=scenarios,
        )
        robustness_summary = {
            **asdict(robustness.monte_carlo),
            "baseline_return": robustness.baseline_return,
            "baseline_max_drawdown": robustness.baseline_max_drawdown,
            "scenario_count": len(robustness.scenarios),
            "stress_degradation_fraction": robustness.stress_degradation_fraction,
            "fragility_score": robustness.fragility_score,
            "fingerprint": robustness.fingerprint,
        }
        stages.append(self._stage(request.job_id, "robustness", robustness_summary))

        risk_result = None
        if request.risk_request is not None:
            risk_result = evaluate_risk_governor(request.risk_request, request.risk_policy)
            stages.append(
                self._stage(
                    request.job_id,
                    "risk_governor",
                    {
                        "action": risk_result.action.value,
                        "reason_count": len(risk_result.reasons),
                        "triggered_constraints": risk_result.triggered_constraints,
                        "fingerprint": risk_result.governor_fingerprint,
                    },
                )
            )
        else:
            stages.append(
                PlatformStageResult(
                    stage="risk_governor",
                    state="SKIPPED",
                    summary={"reason": "no F38 request supplied; paper orders are not permitted"},
                )
            )

        paper_result: dict[str, Any] = {
            "state": "SKIPPED",
            "reason": "no paper orders supplied",
            "live_execution_enabled": False,
        }
        if request.paper_orders:
            assert risk_result is not None
            initial_cash = (
                request.paper_initial_cash
                if request.paper_initial_cash is not None
                else risk_result.equity
            )
            paper = PaperTradingEngine(
                request.paper_account_id,
                initial_cash,
                request.paper_policy,
            )
            # Replay in timestamp order; submit orders before matching quotes at a tie.
            timeline: list[tuple[datetime, int, str, Any]] = []
            timeline.extend(
                (order.submitted_at, 0, "order", order) for order in request.paper_orders
            )
            timeline.extend(
                (quote.timestamp, 1, "quote", quote) for quote in request.paper_quotes
            )
            timeline.sort(
                key=lambda event: (
                    event[0],
                    event[1],
                    getattr(
                        event[3],
                        "idempotency_key",
                        getattr(event[3], "asset_symbol", ""),
                    ),
                )
            )
            for event_time, _, event_type, event_payload in timeline:
                if event_type == "order":
                    paper.submit_order(event_payload, risk_result)
                else:
                    paper.process_quote(event_payload, as_of=event_time)
            final_time = max(event[0] for event in timeline)
            snapshot = paper.snapshot(at=final_time)
            paper_result = {
                "state": "SUCCEEDED",
                "account": snapshot.model_dump(mode="json"),
                "orders": [item.model_dump(mode="json") for item in paper.orders],
                "fills": [item.model_dump(mode="json") for item in paper.fills],
                "audit_events": len(paper.audit_events),
                "audit_chain_verified": paper.verify_audit_chain(),
                "live_execution_enabled": False,
            }
            stages.append(
                self._stage(
                    request.job_id,
                    "paper",
                    {
                        "order_count": len(paper.orders),
                        "fill_count": len(paper.fills),
                        "audit_chain_verified": paper.verify_audit_chain(),
                        "final_equity": snapshot.equity,
                    },
                )
            )
        else:
            stages.append(
                PlatformStageResult(stage="paper", state="SKIPPED", summary=paper_result)
            )

        output = {
            "platform_release": "1.0.0-rc1",
            "job_id": request.job_id,
            "request_fingerprint": fingerprint,
            "dataset": {
                "dataset_id": request.dataset_id,
                "symbol": request.symbol,
                "as_of": request.as_of.isoformat(),
                "bar_count": len(bars),
                "market_fingerprint": _fingerprint(frame.to_dict(orient="records")),
                "quality": {
                    **quality.model_dump(mode="json"),
                    "is_usable": quality.is_usable,
                },
                "features": feature_report.model_dump(mode="json"),
                "feature_last_row": {
                    key: (
                        None
                        if pd.isna(value)
                        else value.isoformat()
                        if hasattr(value, "isoformat")
                        else value
                    )
                    for key, value in featured.tail(1).iloc[0].to_dict().items()
                },
            },
            "algorithm_catalog": catalog,
            "backtest": backtest_summary,
            "walk_forward": walk_forward_result,
            "robustness": {
                **robustness_summary,
                "scenarios": [asdict(item) for item in robustness.scenarios],
            },
            "risk_decision": None if risk_result is None else risk_result.model_dump(mode="json"),
            "paper": paper_result,
            "stages": [item.model_dump(mode="json") for item in stages],
            "reproducibility": {
                "request_fingerprint": fingerprint,
                "feature_pipeline_id": feature_report.pipeline_id,
                "feature_input_fingerprint": feature_report.input_fingerprint,
                "feature_output_fingerprint": feature_report.output_fingerprint,
                "robustness_fingerprint": robustness.fingerprint,
                "strategy_targets_are_caller_supplied": True,
                "seed": request.monte_carlo.seed,
            },
            "safety": {
                "live_execution_enabled": False,
                "risk_gate_required_for_paper_orders": True,
                "does_not_claim_profitability": True,
            },
        }
        self.store.record_event(
            request.job_id,
            "PIPELINE_RESULT_READY",
            {
                "request_fingerprint": fingerprint,
                "feature_pipeline_id": feature_report.pipeline_id,
                "robustness_fingerprint": robustness.fingerprint,
                "risk_action": None if risk_result is None else risk_result.action.value,
                "paper_fill_count": len(paper_result.get("fills", [])),
            },
        )
        return output


__all__ = ["PlatformExecutionError", "PlatformService"]
