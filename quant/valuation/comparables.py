"""Comparable-company valuation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.contracts import ValuationAssumptions, ValuationResult


def peer_multiple_mean(peer_multiples: Sequence[float]) -> float:
    checked = _check_peer_values(peer_multiples)
    return math.fsum(checked) / len(checked)


def peer_multiple_median(peer_multiples: Sequence[float]) -> float:
    checked = sorted(_check_peer_values(peer_multiples))
    middle = len(checked) // 2
    if len(checked) % 2:
        return checked[middle]
    return (checked[middle - 1] + checked[middle]) / 2.0


def comparable_valuation_result(
    multiple: float,
    target_metric: float,
    assumptions: ValuationAssumptions,
    *,
    basis: str = "equity",
    debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: float | None = None,
) -> ValuationResult:
    multiple = _finite(multiple, "multiple")
    target_metric = _positive(target_metric, "target_metric")
    if basis not in {"equity", "enterprise"}:
        raise ValueError("basis must be 'equity' or 'enterprise'")
    if basis == "equity":
        equity_value = multiple * target_metric
        enterprise_value = None
    else:
        enterprise_value = multiple * target_metric
        equity_value = enterprise_value - _non_negative(debt, "debt") + _non_negative(cash, "cash")

    per_share = None
    unit = "currency"
    if shares_outstanding is not None:
        shares = _positive(shares_outstanding, "shares_outstanding")
        per_share = equity_value / shares
        unit = "currency_per_share"

    return ValuationResult(
        method="comparables",
        assumptions_version=assumptions.assumptions_version,
        currency=assumptions.currency,
        basis="equity" if basis == "equity" else "enterprise",
        unit=unit,
        value=equity_value if basis == "equity" else enterprise_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        per_share_value=per_share,
    )


def _check_peer_values(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("peer multiple set cannot be empty")
    checked = tuple(_finite(value, "peer_multiple") for value in values)
    if any(value <= 0.0 for value in checked):
        raise ValueError("peer multiples must be positive")
    return checked


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return value
