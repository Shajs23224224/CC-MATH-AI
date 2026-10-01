"""Deterministic dependence statistics."""

from __future__ import annotations

from collections.abc import Sequence

from quant.math import correlation

from .validation import validate_sample


def rankdata(values: Sequence[float]) -> tuple[float, ...]:
    """Return average ranks, with ties receiving their mean rank."""
    checked = validate_sample(values)
    indexed = sorted(enumerate(checked), key=lambda item: item[1])
    ranks = [0.0] * len(checked)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in range(position, end):
            ranks[indexed[index][0]] = average_rank
        position = end
    return tuple(ranks)


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman rank correlation using average-tie ranks."""
    left_checked = validate_sample(left, minimum=2)
    right_checked = validate_sample(right, minimum=2)
    if len(left_checked) != len(right_checked):
        raise ValueError("left and right must have the same length")
    return correlation(rankdata(left_checked), rankdata(right_checked))


def autocorrelation(values: Sequence[float], lag: int = 1) -> float:
    """Return Pearson autocorrelation at a positive integer lag."""
    checked = validate_sample(values, minimum=3)
    if lag < 1 or lag >= len(checked):
        raise ValueError("lag must be at least 1 and smaller than the sample size")
    left = checked[:-lag]
    right = checked[lag:]
    return correlation(left, right)
