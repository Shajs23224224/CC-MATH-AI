"""F38 deterministic final veto/reduction gate built over F21/F22 and F37."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from itertools import combinations
from typing import Any

from core.contracts import RiskAction, RiskConstraint, RiskEvaluation, RiskMetric
from risk.constraints import evaluate_constraints
from risk.metrics.portfolio import liquidity_risk_metrics
from risk.governor.contracts import (
    GovernorApprovedPosition,
    RiskGovernorConfig,
    RiskGovernorRequest,
    RiskGovernorResult,
)


def _fingerprint(payload: Any) -> str:
    """Hash canonical JSON for deterministic governor replay."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_timestamp(value: str) -> datetime | None:
    """Parse ISO timestamps without silently assuming a timezone."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _risk_constraints(policy: RiskGovernorConfig) -> tuple[RiskConstraint, ...]:
    """Create the fixed F38 policy set; absent required metrics fail closed."""
    return (
        RiskConstraint(
            constraint_id="max-position-weight",
            name="Maximum single-position weight",
            metric_name="max_position_weight",
            limit=policy.max_position_weight,
            hard=True,
            expected_unit="fraction_of_equity",
        ),
        RiskConstraint(
            constraint_id="gross-leverage",
            name="Maximum gross leverage",
            metric_name="gross_leverage",
            limit=policy.max_gross_leverage,
            hard=True,
            expected_unit="multiple_of_equity",
        ),
        RiskConstraint(
            constraint_id="concentration-hhi",
            name="Maximum concentration HHI",
            metric_name="concentration_hhi",
            limit=policy.max_concentration_hhi,
            hard=False,
            expected_unit="index_0_to_1",
        ),
        RiskConstraint(
            constraint_id="sector-weight",
            name="Maximum sector exposure",
            metric_name="max_sector_weight",
            limit=policy.max_sector_weight,
            hard=True,
            expected_unit="fraction_of_equity",
        ),
        RiskConstraint(
            constraint_id="factor-exposure",
            name="Maximum absolute factor exposure",
            metric_name="max_abs_factor_exposure",
            limit=policy.max_abs_factor_exposure,
            hard=False,
            expected_unit="fraction_of_equity",
        ),
        RiskConstraint(
            constraint_id="pairwise-correlation",
            name="Maximum absolute pairwise correlation",
            metric_name="max_abs_correlation",
            limit=policy.max_abs_correlation,
            hard=False,
            expected_unit="absolute_correlation",
        ),
        RiskConstraint(
            constraint_id="drawdown",
            name="Maximum current drawdown",
            metric_name="max_drawdown",
            limit=policy.max_drawdown_fraction,
            hard=True,
            expected_unit="fraction",
        ),
        RiskConstraint(
            constraint_id="daily-loss",
            name="Maximum daily loss",
            metric_name="daily_loss_fraction",
            limit=policy.max_daily_loss_fraction,
            hard=True,
            expected_unit="fraction",
        ),
        RiskConstraint(
            constraint_id="liquidity-days",
            name="Maximum estimated liquidation horizon",
            metric_name="estimated_liquidation_days",
            limit=policy.max_liquidation_days,
            hard=False,
            expected_unit="trading_days",
        ),
        RiskConstraint(
            constraint_id="tail-loss",
            name="Maximum expected shortfall",
            metric_name="max_cvar_fraction",
            limit=policy.max_tail_loss_fraction,
            hard=True,
            expected_unit="fraction_of_notional",
        ),
        RiskConstraint(
            constraint_id="turnover",
            name="Maximum turnover ratio",
            metric_name="turnover_ratio",
            limit=policy.max_turnover_ratio,
            hard=False,
            expected_unit="multiple_of_equity",
        ),
        RiskConstraint(
            constraint_id="data-quality",
            name="Minimum data quality",
            metric_name="data_quality_score",
            limit=policy.min_data_quality_score,
            operator="min",
            hard=True,
            expected_unit="score_0_to_1",
        ),
        RiskConstraint(
            constraint_id="model-quality",
            name="Minimum model quality",
            metric_name="model_quality_score",
            limit=policy.min_model_quality_score,
            operator="min",
            hard=True,
            expected_unit="score_0_to_1",
        ),
    )


def _metric(name: str, value: float, unit: str, request: RiskGovernorRequest, method: str) -> RiskMetric:
    """Make a derived metric with a consistent timestamp and methodology."""
    return RiskMetric(
        name=name,
        value=float(value),
        unit=unit,
        timestamp=request.as_of.isoformat(),
        methodology=method,
    )


def _derived_metrics(
    request: RiskGovernorRequest,
    policy: RiskGovernorConfig,
) -> tuple[RiskMetric, ...]:
    """Calculate exposures from F37 targets and risk context, never from future data."""
    allocations = request.sizing_result.allocations
    contexts = {asset.asset_symbol: asset for asset in request.assets}
    equity = request.sizing_result.equity
    notionals = {item.asset_symbol: item.target_notional for item in allocations}
    weights = {item.asset_symbol: item.target_weight for item in allocations}
    active_symbols = tuple(
        item.asset_symbol for item in allocations if item.target_quantity > 0.0
    )
    gross_notional = math.fsum(notionals.values())

    metrics: list[RiskMetric] = []
    metrics.append(
        _metric(
            "max_position_weight",
            max(weights.values(), default=0.0),
            "fraction_of_equity",
            request,
            "Maximum F37 target notional divided by account equity.",
        )
    )
    metrics.append(
        _metric(
            "gross_leverage",
            gross_notional / equity,
            "multiple_of_equity",
            request,
            "Sum of long-only F37 target notionals divided by account equity.",
        )
    )
    concentration = (
        math.fsum((notional / gross_notional) ** 2 for notional in notionals.values())
        if gross_notional > 0.0
        else 0.0
    )
    metrics.append(
        _metric(
            "concentration_hhi",
            concentration,
            "index_0_to_1",
            request,
            "Sum of squared target-notional shares of gross invested notional.",
        )
    )

    sector_notionals: dict[str, float] = {}
    for symbol, notional in notionals.items():
        sector = contexts[symbol].sector
        sector_notionals[sector] = sector_notionals.get(sector, 0.0) + notional
    max_sector_weight = max(sector_notionals.values(), default=0.0) / equity
    metrics.append(
        _metric(
            "max_sector_weight",
            max_sector_weight,
            "fraction_of_equity",
            request,
            "Largest sector's summed F37 target notional divided by account equity.",
        )
    )

    factor_sets = [set(contexts[symbol].factor_loadings) for symbol in active_symbols]
    if factor_sets and factor_sets[0] and all(factors == factor_sets[0] for factors in factor_sets):
        factor_exposures = {
            factor: math.fsum(
                weights[symbol] * contexts[symbol].factor_loadings[factor]
                for symbol in active_symbols
            )
            for factor in sorted(factor_sets[0])
        }
        max_factor_exposure = max(
            (abs(value) for value in factor_exposures.values()),
            default=0.0,
        )
        metrics.append(
            _metric(
                "max_abs_factor_exposure",
                max_factor_exposure,
                "fraction_of_equity",
                request,
                "Maximum absolute equity-weighted factor loading across active positions.",
            )
        )

    pair_map = {pair.pair_key: pair.correlation for pair in request.pairwise_correlations}
    active_pairs = tuple(combinations(active_symbols, 2))
    missing_pairs = [
        tuple(sorted(pair)) for pair in active_pairs if tuple(sorted(pair)) not in pair_map
    ]
    if not active_pairs:
        max_abs_correlation = 0.0
    elif not missing_pairs:
        max_abs_correlation = max(
            abs(pair_map[tuple(sorted(pair))]) for pair in active_pairs
        )
    else:
        max_abs_correlation = None
    if max_abs_correlation is not None:
        metrics.append(
            _metric(
                "max_abs_correlation",
                max_abs_correlation,
                "absolute_correlation",
                request,
                "Maximum absolute F23 pairwise correlation among non-zero target positions.",
            )
        )

    liquidation_days: list[float] = []
    for allocation in allocations:
        if allocation.target_quantity <= 0.0:
            continue
        context = contexts[allocation.asset_symbol]
        liquidity = liquidity_risk_metrics(
            allocation.asset_symbol,
            allocation.target_notional,
            context.average_daily_dollar_volume,
            max_participation_rate=policy.max_liquidity_participation,
        )
        liquidation_days.append(liquidity.estimated_liquidation_days)
    if liquidation_days:
        metrics.append(
            _metric(
                "estimated_liquidation_days",
                max(liquidation_days),
                "trading_days",
                request,
                "Maximum F21 estimated liquidation days using ADV and the configured participation cap.",
            )
        )

    traded_notional = math.fsum(
        abs(notionals[item.asset_symbol] - contexts[item.asset_symbol].current_notional)
        for item in allocations
    )
    metrics.append(
        _metric(
            "turnover_ratio",
            traded_notional / equity,
            "multiple_of_equity",
            request,
            "Sum of absolute target-minus-current notional divided by account equity.",
        )
    )
    return tuple(metrics)


def _freshness_failures(
    request: RiskGovernorRequest,
    policy: RiskGovernorConfig,
) -> tuple[str, ...]:
    """Fail closed on stale/future F37, ADV, dependence, or risk evidence."""
    failures: list[str] = []
    cutoff_seconds = policy.max_input_age_seconds

    def check(label: str, timestamp: datetime) -> None:
        age = (request.as_of - timestamp).total_seconds()
        if age < 0.0:
            failures.append(f"{label} timestamp is later than governor as_of")
        elif age > cutoff_seconds:
            failures.append(
                f"{label} timestamp is stale by {age:.0f}s "
                f"(maximum {cutoff_seconds}s)"
            )

    check("F37 sizing", request.sizing_result.as_of)
    check("market data", request.market_data_as_of)
    check("risk/correlation snapshot", request.risk_snapshot_as_of)
    for metric in request.risk_metrics:
        timestamp = _parse_timestamp(metric.timestamp)
        if timestamp is None:
            failures.append(f"risk metric {metric.name!r} has an invalid or timezone-free timestamp")
        else:
            check(f"risk metric {metric.name!r}", timestamp)
    return tuple(failures)


def _blocked_evaluation(
    reasons: tuple[str, ...],
    metrics: tuple[RiskMetric, ...] = (),
    triggered: tuple[str, ...] = (),
) -> RiskEvaluation:
    """Construct a hard veto with zero exposure."""
    return RiskEvaluation(
        action=RiskAction.BLOCK,
        reasons=reasons,
        metrics=metrics,
        triggered_constraints=triggered,
        reduction_factor=0.0,
    )


def evaluate_risk_governor(
    request: RiskGovernorRequest,
    config: RiskGovernorConfig | None = None,
) -> RiskGovernorResult:
    """Apply F38 final ALLOW/REDUCE/BLOCK gate to the F37 sized portfolio.

    This is a risk decision only. It never sends an order or converts a BLOCK
    into permission; missing required evidence, stale inputs, failed controls,
    or an unrepresentable reduction all fail closed.
    """
    policy = config or RiskGovernorConfig()
    allocations = request.sizing_result.allocations
    contexts = {asset.asset_symbol: asset for asset in request.assets}
    freshness_failures = _freshness_failures(request, policy)

    if request.sizing_result.status == "ZERO_SIZE":
        evaluation = _blocked_evaluation(
            ("F37 returned ZERO_SIZE; no positive exposure can be approved.",),
            tuple(request.risk_metrics),
            ("upstream-zero-size",),
        )
    elif freshness_failures:
        evaluation = _blocked_evaluation(
            freshness_failures,
            tuple(request.risk_metrics),
            tuple(f"stale-input-{index + 1}" for index in range(len(freshness_failures))),
        )
    else:
        derived_metrics = _derived_metrics(request, policy)
        metrics = (*derived_metrics, *sorted(request.risk_metrics, key=lambda item: item.name))
        try:
            evaluation = evaluate_constraints(
                metrics,
                _risk_constraints(policy),
                reduction_factor=policy.reduction_factor,
            )
        except ValueError as error:
            evaluation = _blocked_evaluation(
                (f"risk metric evaluation failed closed: {error}",),
                tuple(metrics),
                ("risk-evaluator-error",),
            )

    positions: list[GovernorApprovedPosition] = []
    for allocation in allocations:
        context = contexts[allocation.asset_symbol]
        source_quantity = allocation.target_quantity
        if evaluation.action == RiskAction.BLOCK:
            approved_quantity = 0.0
            approved_notional = 0.0
            status = "BLOCKED"
            rationale = "F38 BLOCK: no downstream module may override this veto."
        else:
            factor = evaluation.reduction_factor
            if factor == 1.0:
                approved_quantity = source_quantity
            else:
                lots = math.floor(
                    source_quantity * factor / context.lot_size + 1e-12
                )
                approved_quantity = max(0.0, lots * context.lot_size)
            if source_quantity > 0.0:
                unit_price = allocation.target_notional / source_quantity
                approved_notional = approved_quantity * unit_price
            else:
                approved_notional = 0.0
            if approved_quantity <= 0.0:
                status = "ZERO_SIZE"
                rationale = "The policy reduction and lot rounding leave no positive quantity."
            elif approved_quantity < source_quantity:
                status = "REDUCED"
                rationale = "F38 reduced F37 quantity and rounded down to the configured lot size."
            else:
                status = "APPROVED"
                rationale = "F38 controls passed; the F37 quantity remains within configured limits."

        approved_weight = approved_notional / request.sizing_result.equity
        positions.append(
            GovernorApprovedPosition(
                asset_symbol=allocation.asset_symbol,
                source_quantity=source_quantity,
                approved_quantity=approved_quantity,
                approved_notional=approved_notional,
                approved_weight=approved_weight,
                lot_size=context.lot_size,
                status=status,
                rationale=rationale,
            )
        )

    if evaluation.action == RiskAction.REDUCE and not any(
        item.approved_quantity > 0.0 for item in positions
    ):
        evaluation = _blocked_evaluation(
            (*evaluation.reasons, "Risk reduction rounded every position to zero; fail closed."),
            evaluation.metrics,
            (*evaluation.triggered_constraints, "reduction-left-no-executable-lots"),
        )
        positions = [
            item.model_copy(
                update={
                    "approved_quantity": 0.0,
                    "approved_notional": 0.0,
                    "approved_weight": 0.0,
                    "status": "BLOCKED",
                    "rationale": "F38 BLOCK: reduction left no positive lot-sized quantity.",
                }
            )
            for item in positions
        ]

    positions_tuple = tuple(positions)
    audit_payload = {
        "request": request.model_dump(mode="json"),
        "config": policy.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
        "positions": [item.model_dump(mode="json") for item in positions_tuple],
    }
    return RiskGovernorResult(
        action=evaluation.action,
        reduction_factor=evaluation.reduction_factor,
        reasons=evaluation.reasons,
        triggered_constraints=evaluation.triggered_constraints,
        metrics=evaluation.metrics,
        positions=positions_tuple,
        equity=request.sizing_result.equity,
        total_approved_notional=math.fsum(item.approved_notional for item in positions_tuple),
        total_approved_weight=math.fsum(item.approved_weight for item in positions_tuple),
        config_id=policy.config_id,
        config_version=policy.version,
        as_of=request.as_of,
        data_snapshot_id=request.data_snapshot_id,
        risk_snapshot_id=request.risk_snapshot_id,
        model_snapshot_id=request.model_snapshot_id,
        sizing_fingerprint=request.sizing_result.sizing_fingerprint,
        governor_fingerprint=_fingerprint(audit_payload),
    )


__all__ = ["evaluate_risk_governor"]
