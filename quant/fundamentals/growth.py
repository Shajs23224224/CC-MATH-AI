"""Point-in-time fundamental growth calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from core.contracts import FundamentalGrowthPoint, FundamentalSnapshot

_SUPPORTED_METRICS = frozenset({"revenue", "ebitda", "eps", "free_cash_flow"})


def growth_rate(current: float, previous: float) -> float:
    """Return simple growth (current / previous - 1) with explicit zero handling."""
    if not math.isfinite(current) or not math.isfinite(previous):
        raise ValueError("growth inputs must be finite")
    if previous == 0.0:
        raise ValueError("previous value cannot be zero for growth")
    return current / previous - 1.0


def fundamental_growth(
    snapshots: Sequence[FundamentalSnapshot],
    metric: str,
    *,
    as_of: date,
) -> tuple[FundamentalGrowthPoint, ...]:
    """Return growth observations using only information reported by as_of.

    For duplicate accounting periods, only the latest report available by
    as_of is used. No future restatement can enter the result.
    """
    if metric not in _SUPPORTED_METRICS:
        allowed = ", ".join(sorted(_SUPPORTED_METRICS))
        raise ValueError(f"unsupported growth metric {metric!r}; expected one of: {allowed}")
    if not snapshots:
        raise ValueError("at least one snapshot is required")

    first_asset = snapshots[0].asset
    if any(snapshot.asset != first_asset for snapshot in snapshots[1:]):
        raise ValueError("all snapshots must belong to the same asset")

    available = [snapshot for snapshot in snapshots if snapshot.reported_at <= as_of]
    if not available:
        raise ValueError("no snapshots are available at the requested as_of date")

    selected: dict[date, FundamentalSnapshot] = {}
    for snapshot in available:
        prior = selected.get(snapshot.period_end)
        if prior is not None and prior.reported_at == snapshot.reported_at:
            raise ValueError("duplicate period_end/reported_at snapshots are not allowed")
        if prior is None or snapshot.reported_at > prior.reported_at:
            selected[snapshot.period_end] = snapshot

    ordered = sorted(selected.values(), key=lambda item: item.period_end)
    points: list[FundamentalGrowthPoint] = []
    previous_value: float | None = None

    for snapshot in ordered:
        value = getattr(snapshot, metric)
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{metric} must be finite when present")
        growth: float | None = None
        if value is not None and previous_value is not None:
            growth = growth_rate(float(value), previous_value)
        points.append(
            FundamentalGrowthPoint(
                asset=snapshot.asset,
                metric=metric,
                period_end=snapshot.period_end,
                reported_at=snapshot.reported_at,
                value=None if value is None else float(value),
                growth=growth,
            )
        )
        previous_value = None if value is None else float(value)

    return tuple(points)
