"""Fail-safe evaluation of named risk metrics against constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.contracts import (
    Portfolio,
    RiskAction,
    RiskConstraint,
    RiskEvaluation,
    RiskMetric,
)
from risk.metrics.portfolio import portfolio_risk_metrics

_CANONICAL_UNITS = {
    "gross_leverage": "multiple_of_equity",
    "net_exposure_ratio": "multiple_of_equity",
    "cash_weight": "fraction_of_equity",
    "concentration_hhi": "index_0_to_1",
    "largest_position_weight": "fraction_of_gross_exposure",
    "drawdown": "fraction",
    "max_drawdown": "fraction",
    "liquidation_days": "trading_days",
    "estimated_liquidation_days": "trading_days",
}


def evaluate_constraints(
    metrics: Sequence[RiskMetric],
    constraints: Sequence[RiskConstraint],
    *,
    reduction_factor: float = 0.5,
) -> RiskEvaluation:
    """Apply constraints, blocking on hard breaches or unavailable required metrics.

    Missing metrics fail closed because the evaluator cannot prove the portfolio
    is within a required limit. Soft breaches yield REDUCE with a deterministic
    allocation multiplier; any hard breach takes precedence and yields BLOCK.
    """
    if not math.isfinite(reduction_factor) or not 0.0 < reduction_factor < 1.0:
        raise ValueError("reduction_factor must be finite and strictly between 0 and 1")

    metric_items = tuple(metrics)
    constraint_items = tuple(constraints)
    if not constraint_items:
        return RiskEvaluation(
            action=RiskAction.BLOCK,
            reasons=("no risk constraints configured; evaluation fails closed",),
            metrics=metric_items,
            reduction_factor=0.0,
        )

    metric_by_name: dict[str, RiskMetric] = {}
    for metric in metric_items:
        if metric.name in metric_by_name:
            raise ValueError(f"duplicate risk metric name: {metric.name}")
        metric_by_name[metric.name] = metric

    seen_constraint_ids: set[str] = set()
    for constraint in constraint_items:
        if constraint.constraint_id in seen_constraint_ids:
            raise ValueError(f"duplicate risk constraint id: {constraint.constraint_id}")
        seen_constraint_ids.add(constraint.constraint_id)

    reasons: list[str] = []
    triggered: list[str] = []
    hard_breach = False
    soft_breach = False

    for constraint in constraint_items:
        matched_metric = metric_by_name.get(constraint.metric_name)
        if matched_metric is None:
            hard_breach = True
            triggered.append(constraint.constraint_id)
            reasons.append(
                f"{constraint.constraint_id}: required metric "
                f"{constraint.metric_name!r} is missing; evaluation fails closed"
            )
            continue

        expected_unit = constraint.expected_unit or _CANONICAL_UNITS.get(
            constraint.metric_name
        )
        if expected_unit is not None and matched_metric.unit != expected_unit:
            hard_breach = True
            triggered.append(constraint.constraint_id)
            reasons.append(
                f"{constraint.constraint_id}: metric {constraint.metric_name!r} has unit "
                f"{matched_metric.unit!r}; expected {expected_unit!r}; evaluation fails closed"
            )
            continue

        breached = (
            matched_metric.value > constraint.limit
            if constraint.operator == "max"
            else matched_metric.value < constraint.limit
        )
        if not breached:
            continue

        triggered.append(constraint.constraint_id)
        reasons.append(
            f"{constraint.constraint_id}: {constraint.metric_name}={matched_metric.value:g} "
            f"breaches {constraint.operator} limit {constraint.limit:g}"
        )
        if constraint.hard:
            hard_breach = True
        else:
            soft_breach = True

    if hard_breach:
        action = RiskAction.BLOCK
        factor = 0.0
    elif soft_breach:
        action = RiskAction.REDUCE
        factor = reduction_factor
    else:
        action = RiskAction.ALLOW
        factor = 1.0

    return RiskEvaluation(
        action=action,
        reasons=tuple(reasons),
        metrics=metric_items,
        triggered_constraints=tuple(triggered),
        reduction_factor=factor,
    )


def evaluate_portfolio_risk(
    portfolio: Portfolio,
    constraints: Sequence[RiskConstraint],
    *,
    additional_metrics: Sequence[RiskMetric] = (),
    reduction_factor: float = 0.5,
) -> RiskEvaluation:
    """Evaluate portfolio exposures and caller-supplied risk measures together.

    Caller-supplied metrics can include drawdown, beta, liquidity and turnover.
    Metric names must be unique. No decision is sent to a broker or execution
    adapter by this function.
    """
    metrics = (*portfolio_risk_metrics(portfolio), *tuple(additional_metrics))
    return evaluate_constraints(metrics, constraints, reduction_factor=reduction_factor)
