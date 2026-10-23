"""Shared stable probability transform used by calibration and decision inference."""

from __future__ import annotations

import math


def class_probabilities(
    score: float,
    temperature: float,
    hold_bias: float,
) -> tuple[float, float, float]:
    """Return temperature-scaled BUY/HOLD/SELL softmax probabilities."""
    if not math.isfinite(score) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("score must be finite and temperature must be finite and positive")
    if not math.isfinite(hold_bias):
        raise ValueError("hold_bias must be finite")
    buy_logit = score / temperature
    sell_logit = -score / temperature
    max_logit = max(buy_logit, hold_bias, sell_logit)
    buy_weight = math.exp(buy_logit - max_logit)
    hold_weight = math.exp(hold_bias - max_logit)
    sell_weight = math.exp(sell_logit - max_logit)
    denominator = math.fsum((buy_weight, hold_weight, sell_weight))
    return (
        buy_weight / denominator,
        hold_weight / denominator,
        sell_weight / denominator,
    )
