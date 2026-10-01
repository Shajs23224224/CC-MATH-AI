from __future__ import annotations

import math
from collections.abc import Sequence

from .validation import validate_ohlcv, validate_series, validate_window


def _sma(values: tuple[float, ...], window: int) -> tuple[float | None, ...]:
    return tuple(
        None if i + 1 < window else sum(values[i + 1 - window : i + 1]) / window
        for i in range(len(values))
    )


def sma(values: Sequence[float], window: int) -> tuple[float | None, ...]:
    checked = validate_series(values, window)
    return _sma(checked, validate_window(window))


def ema(values: Sequence[float], window: int) -> tuple[float | None, ...]:
    checked = validate_series(values, window)
    period = validate_window(window)
    alpha = 2.0 / (period + 1.0)
    result: list[float | None] = [None] * len(checked)
    current = sum(checked[:period]) / period
    result[period - 1] = current
    for i in range(period, len(checked)):
        current = alpha * checked[i] + (1.0 - alpha) * current
        result[i] = current
    return tuple(result)


def wma(values: Sequence[float], window: int) -> tuple[float | None, ...]:
    checked = validate_series(values, window)
    period = validate_window(window)
    denominator = period * (period + 1) / 2
    return tuple(
        None
        if i + 1 < period
        else sum(
            value * weight
            for value, weight in zip(
                checked[i + 1 - period : i + 1], range(1, period + 1), strict=True
            )
        )
        / denominator
        for i in range(len(checked))
    )


def momentum(values: Sequence[float], period: int = 1) -> tuple[float | None, ...]:
    checked = validate_series(values, period + 1)
    lag = validate_window(period)
    return tuple(None if i < lag else checked[i] - checked[i - lag] for i in range(len(checked)))


def rate_of_change(values: Sequence[float], period: int = 1) -> tuple[float | None, ...]:
    checked = validate_series(values, period + 1)
    lag = validate_window(period)
    if any(value == 0 for value in checked):
        raise ValueError("ROC requires non-zero prices")
    return tuple(
        None if i < lag else checked[i] / checked[i - lag] - 1.0 for i in range(len(checked))
    )


def rsi(values: Sequence[float], window: int = 14) -> tuple[float | None, ...]:
    checked = validate_series(values, window + 1)
    period = validate_window(window)
    gains = [max(checked[i] - checked[i - 1], 0.0) for i in range(1, len(checked))]
    losses = [max(checked[i - 1] - checked[i], 0.0) for i in range(1, len(checked))]
    result: list[float | None] = [None] * len(checked)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def rsi_value(gain: float, loss: float) -> float:
        if loss == 0.0:
            return 100.0 if gain > 0.0 else 50.0
        relative_strength = gain / loss
        return 100.0 - 100.0 / (1.0 + relative_strength)

    result[period] = rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(checked)):
        avg_gain = ((period - 1) * avg_gain + gains[i - 1]) / period
        avg_loss = ((period - 1) * avg_loss + losses[i - 1]) / period
        result[i] = rsi_value(avg_gain, avg_loss)
    return tuple(result)


def macd(
    values: Sequence[float],
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...], tuple[float | None, ...]]:
    fast = validate_window(fast_window)
    slow = validate_window(slow_window)
    signal = validate_window(signal_window)
    if fast >= slow:
        raise ValueError("fast_window must be smaller than slow_window")
    checked = validate_series(values, slow + signal - 1)
    fast_ema = ema(checked, fast)
    slow_ema = ema(checked, slow)
    line: list[float | None] = [None] * len(checked)
    for i, (fast_value, slow_value) in enumerate(zip(fast_ema, slow_ema, strict=True)):
        if fast_value is not None and slow_value is not None:
            line[i] = fast_value - slow_value
    valid_line = tuple(value for value in line if value is not None)
    signal_values = ema(valid_line, signal)
    signal_line: list[float | None] = [None] * len(checked)
    histogram: list[float | None] = [None] * len(checked)
    start = slow - 1
    for j, signal_value in enumerate(signal_values):
        i = start + j
        signal_line[i] = signal_value
        if signal_value is not None and line[i] is not None:
            histogram[i] = line[i] - signal_value
    return tuple(line), tuple(signal_line), tuple(histogram)


def atr(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int = 14,
) -> tuple[float | None, ...]:
    period = validate_window(window)
    checked_high, checked_low, checked_close, _ = validate_ohlcv(high, low, close)
    if len(checked_close) < period:
        raise ValueError("at least window observations are required")
    true_ranges = [
        checked_high[0] - checked_low[0],
        *(
            max(
                checked_high[i] - checked_low[i],
                abs(checked_high[i] - checked_close[i - 1]),
                abs(checked_low[i] - checked_close[i - 1]),
            )
            for i in range(1, len(checked_close))
        ),
    ]
    result: list[float | None] = [None] * len(checked_close)
    current = sum(true_ranges[:period]) / period
    result[period - 1] = current
    for i in range(period, len(true_ranges)):
        current = ((period - 1) * current + true_ranges[i]) / period
        result[i] = current
    return tuple(result)


def bollinger_bands(
    values: Sequence[float],
    window: int = 20,
    deviations: float = 2.0,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...], tuple[float | None, ...]]:
    period = validate_window(window)
    checked = validate_series(values, period)
    if deviations < 0 or not math.isfinite(deviations):
        raise ValueError("deviations must be finite and non-negative")
    middle = _sma(checked, period)
    lower: list[float | None] = [None] * len(checked)
    upper: list[float | None] = [None] * len(checked)
    for i in range(period - 1, len(checked)):
        sample = checked[i + 1 - period : i + 1]
        mean = middle[i]
        if mean is None:
            continue
        variance = sum((value - mean) ** 2 for value in sample) / period
        std = math.sqrt(variance)
        lower[i] = mean - deviations * std
        upper[i] = mean + deviations * std
    return middle, tuple(upper), tuple(lower)


def stochastic_oscillator(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int = 14,
    smooth: int = 3,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    period = validate_window(window)
    smoothing = validate_window(smooth)
    checked_high, checked_low, checked_close, _ = validate_ohlcv(high, low, close)
    if len(checked_close) < period + smoothing - 1:
        raise ValueError("at least window + smooth - 1 observations are required")
    k: list[float | None] = [None] * len(checked_close)
    for i in range(period - 1, len(checked_close)):
        highest = max(checked_high[i + 1 - period : i + 1])
        lowest = min(checked_low[i + 1 - period : i + 1])
        if highest == lowest:
            k[i] = 50.0
        else:
            k[i] = 100.0 * (checked_close[i] - lowest) / (highest - lowest)
    valid_k = tuple(value for value in k if value is not None)
    d_valid = _sma(valid_k, smoothing)
    d: list[float | None] = [None] * len(checked_close)
    start = period - 1
    for j, value in enumerate(d_valid):
        d[start + j] = value
    return tuple(k), tuple(d)


def williams_r(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int = 14,
) -> tuple[float | None, ...]:
    period = validate_window(window)
    checked_high, checked_low, checked_close, _ = validate_ohlcv(high, low, close)
    if len(checked_close) < period:
        raise ValueError("at least window observations are required")
    result: list[float | None] = [None] * len(checked_close)
    for i in range(period - 1, len(checked_close)):
        highest = max(checked_high[i + 1 - period : i + 1])
        lowest = min(checked_low[i + 1 - period : i + 1])
        if highest == lowest:
            result[i] = -50.0
        else:
            result[i] = -100.0 * (highest - checked_close[i]) / (highest - lowest)
    return tuple(result)


def adx_dmi(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int = 14,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...], tuple[float | None, ...]]:
    period = validate_window(window)
    checked_high, checked_low, checked_close, _ = validate_ohlcv(high, low, close)
    n = len(checked_close)
    if n < 2 * period:
        raise ValueError("ADX/DMI requires at least 2 * window observations")
    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        tr[i] = max(
            checked_high[i] - checked_low[i],
            abs(checked_high[i] - checked_close[i - 1]),
            abs(checked_low[i] - checked_close[i - 1]),
        )
        up_move = checked_high[i] - checked_high[i - 1]
        down_move = checked_low[i - 1] - checked_low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    atr_values: list[float | None] = [None] * n
    plus_di: list[float | None] = [None] * n
    minus_di: list[float | None] = [None] * n
    dx: list[float | None] = [None] * n
    smoothed_tr = sum(tr[1 : period + 1])
    smoothed_plus = sum(plus_dm[1 : period + 1])
    smoothed_minus = sum(minus_dm[1 : period + 1])
    for i in range(period, n):
        if i > period:
            smoothed_tr = smoothed_tr - smoothed_tr / period + tr[i]
            smoothed_plus = smoothed_plus - smoothed_plus / period + plus_dm[i]
            smoothed_minus = smoothed_minus - smoothed_minus / period + minus_dm[i]
        atr_values[i] = smoothed_tr / period
        if smoothed_tr == 0:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
            dx[i] = 0.0
        else:
            plus_di[i] = 100.0 * smoothed_plus / smoothed_tr
            minus_di[i] = 100.0 * smoothed_minus / smoothed_tr
            denominator = plus_di[i] + minus_di[i]
            dx[i] = (
                0.0
                if denominator == 0
                else 100.0 * abs(plus_di[i] - minus_di[i]) / denominator
            )
    adx: list[float | None] = [None] * n
    valid_dx = tuple(value for value in dx if value is not None)
    current = sum(valid_dx[:period]) / period
    adx_index = 2 * period - 1
    adx[adx_index] = current
    for j in range(period, len(valid_dx)):
        current = ((period - 1) * current + valid_dx[j]) / period
        index = period + j
        if index < n:
            adx[index] = current
    return tuple(plus_di), tuple(minus_di), tuple(adx)


def donchian_channels(
    high: Sequence[float],
    low: Sequence[float],
    window: int = 20,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...], tuple[float | None, ...]]:
    period = validate_window(window)
    checked_high = validate_series(high, period)
    checked_low = validate_series(low, period)
    if len(checked_high) != len(checked_low):
        raise ValueError("high and low must have equal length")
    for high_value, low_value in zip(checked_high, checked_low, strict=True):
        if high_value < low_value:
            raise ValueError("high must be >= low")
    upper: list[float | None] = [None] * len(checked_high)
    lower: list[float | None] = [None] * len(checked_high)
    middle: list[float | None] = [None] * len(checked_high)
    for i in range(period - 1, len(checked_high)):
        high_value = max(checked_high[i + 1 - period : i + 1])
        low_value = min(checked_low[i + 1 - period : i + 1])
        upper[i] = high_value
        lower[i] = low_value
        middle[i] = (high_value + low_value) / 2.0
    return tuple(upper), tuple(middle), tuple(lower)


def vwap(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
) -> tuple[float, ...]:
    checked_high, checked_low, checked_close, checked_volume = validate_ohlcv(
        high, low, close, volume
    )
    assert checked_volume is not None
    cumulative_volume = 0.0
    cumulative_value = 0.0
    result: list[float] = []
    for high_value, low_value, close_value, volume_value in zip(
        checked_high, checked_low, checked_close, checked_volume, strict=True
    ):
        cumulative_volume += volume_value
        cumulative_value += ((high_value + low_value + close_value) / 3.0) * volume_value
        result.append(
            (high_value + low_value + close_value) / 3.0
            if cumulative_volume == 0
            else cumulative_value / cumulative_volume
        )
    return tuple(result)


def volume_change(volume: Sequence[float], period: int = 1) -> tuple[float | None, ...]:
    checked = validate_series(volume, period + 1)
    lag = validate_window(period)
    return tuple(None if i < lag else checked[i] - checked[i - lag] for i in range(len(checked)))
