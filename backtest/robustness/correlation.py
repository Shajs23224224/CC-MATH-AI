"""Correlation-matrix stress tests for synchronized multi-asset return panels."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite, sqrt
from statistics import mean

from backtest.robustness.contracts import CorrelationStressResult


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    centered_left = tuple(value - left_mean for value in left)
    centered_right = tuple(value - right_mean for value in right)
    left_sum = sum(value * value for value in centered_left)
    right_sum = sum(value * value for value in centered_right)
    if left_sum <= 0 or right_sum <= 0:
        raise ValueError("correlation is undefined for a constant return series")
    cross = sum(a * b for a, b in zip(centered_left, centered_right, strict=True))
    return max(-1.0, min(1.0, cross / sqrt(left_sum * right_sum)))


def stress_correlation_matrix(
    returns_by_asset: Mapping[str, Sequence[float]],
    shock: float,
) -> CorrelationStressResult:
    """Blend correlations toward +1 or the minimum feasible equicorrelation.

    shock is in [-1, 1]. Positive values blend toward a matrix of +1 correlations;
    negative values blend toward the minimum equicorrelation (-1/(n-1) off-diagonal);
    zero leaves the observed matrix unchanged. Convex blending preserves positive
    semidefiniteness up to floating-point rounding.

    F39 backtests a single OHLC stream, so this is a matrix diagnostic, not portfolio PnL.
    """
    if not isfinite(shock) or not -1.0 <= shock <= 1.0:
        raise ValueError("correlation shock must be finite and in [-1, 1]")
    if not isinstance(returns_by_asset, Mapping) or len(returns_by_asset) < 2:
        raise ValueError("at least two aligned asset return series are required")

    asset_names = tuple(sorted(returns_by_asset))
    if any(not isinstance(name, str) or not name.strip() for name in asset_names):
        raise ValueError("asset names must be non-empty strings")
    series = [tuple(returns_by_asset[name]) for name in asset_names]
    if any(len(values) < 2 for values in series):
        raise ValueError("each asset series must contain at least two observations")
    if any(not all(isfinite(value) for value in values) for values in series):
        raise ValueError("asset returns must be finite")
    lengths = {len(values) for values in series}
    if len(lengths) != 1:
        raise ValueError("asset return series must be aligned and have equal lengths")

    count = len(asset_names)
    baseline = tuple(
        tuple(
            1.0 if row == column else _pearson(series[row], series[column])
            for column in range(count)
        )
        for row in range(count)
    )
    alpha = abs(shock)
    minimum_correlation = -1.0 / (count - 1)
    stressed_rows: list[tuple[float, ...]] = []
    for row in range(count):
        values: list[float] = []
        for column in range(count):
            observed = baseline[row][column]
            if row == column:
                values.append(1.0)
            elif shock > 0:
                values.append((1.0 - alpha) * observed + alpha)
            elif shock < 0:
                values.append((1.0 - alpha) * observed + alpha * minimum_correlation)
            else:
                values.append(observed)
        stressed_rows.append(tuple(values))
    stressed = tuple(stressed_rows)

    payload = {
        "asset_names": asset_names,
        "observations": next(iter(lengths)),
        "shock": shock,
        "returns_by_asset": {name: series[index] for index, name in enumerate(asset_names)},
        "baseline_matrix": baseline,
        "stressed_matrix": stressed,
    }
    fingerprint = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return CorrelationStressResult(
        asset_names=asset_names,
        observations=next(iter(lengths)),
        shock=shock,
        baseline_matrix=baseline,
        stressed_matrix=stressed,
        fingerprint=fingerprint,
    )
