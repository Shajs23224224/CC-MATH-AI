from datetime import UTC, datetime, timedelta

import pytest

from core.contracts.technical import TechnicalIndicatorSeries
from quant.technical import (
    adx_dmi,
    atr,
    bollinger_bands,
    donchian_channels,
    ema,
    macd,
    momentum,
    rate_of_change,
    rsi,
    sma,
    stochastic_oscillator,
    volume_change,
    vwap,
    williams_r,
    wma,
)


def test_technical_series_preserves_alignment() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=i) for i in range(4))
    series = TechnicalIndicatorSeries(
        name="sma_2",
        timestamps=timestamps,
        values=(None, 1.5, 2.5, 3.5),
    )
    assert series.timestamps == timestamps
    assert series.values[0] is None


def test_sma_ema_wma_are_deterministic() -> None:
    values = (1.0, 2.0, 3.0, 4.0)
    assert sma(values, 2) == (None, 1.5, 2.5, 3.5)
    assert ema(values, 2) == (None, 1.5, 2.5, 3.5)
    assert wma(values, 2) == (None, 5 / 3, 8 / 3, 11 / 3)


def test_momentum_and_roc() -> None:
    values = (100.0, 105.0, 110.0)
    assert momentum(values, 1) == (None, 5.0, 5.0)
    result = rate_of_change(values, 1)
    assert result[0] is None
    assert result[1] == pytest.approx(0.05)
    assert result[2] == pytest.approx(110 / 105 - 1)


def test_rsi_uptrend() -> None:
    result = rsi(tuple(float(i) for i in range(1, 17)), 14)
    assert result[:14] == (None,) * 14
    assert result[14] == 100.0


def test_macd_alignment_and_warmup() -> None:
    values = tuple(float(i) for i in range(1, 40))
    line, signal, histogram = macd(values, 3, 5, 2)
    assert len(line) == len(signal) == len(histogram) == len(values)
    assert all(value is None for value in line[:4])
    assert all(value is None for value in signal[:5])


def test_atr_bollinger_stochastic_and_williams() -> None:
    high = (10.0, 11.0, 12.0, 13.0, 14.0)
    low = (8.0, 9.0, 10.0, 11.0, 12.0)
    close = (9.0, 10.0, 11.0, 12.0, 13.0)
    atr_values = atr(high, low, close, 3)
    assert atr_values[2] == 2.0
    middle, upper, lower = bollinger_bands(close, 3, 2)
    assert middle[2] == 10.0
    assert upper[2] > middle[2] > lower[2]
    k, d = stochastic_oscillator(high, low, close, 3, 2)
    assert k[2] == 75.0
    assert d[3] is not None
    assert williams_r(high, low, close, 3)[2] == -25.0


def test_adx_dmi_and_donchian_vwap_volume() -> None:
    high = (10.0, 11.0, 12.0, 13.0, 14.0, 15.0)
    low = (8.0, 9.0, 10.0, 11.0, 12.0, 13.0)
    close = (9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
    volume = (100.0, 110.0, 120.0, 130.0, 140.0, 150.0)
    plus_di, minus_di, adx = adx_dmi(high, low, close, 3)
    assert plus_di[3] is not None and minus_di[3] == 0.0
    assert adx[-1] is not None
    upper, middle, lower = donchian_channels(high, low, 3)
    assert upper[2] == 12.0 and lower[2] == 8.0 and middle[2] == 10.0
    assert vwap(high, low, close, volume)[0] == pytest.approx(9.0)
    assert volume_change(volume, 2)[2] == 20.0


def test_no_lookahead() -> None:
    prefix = (10.0, 11.0, 12.0, 13.0, 14.0)
    extended = prefix + (1000.0,)
    assert sma(prefix, 3)[-1] == sma(extended, 3)[len(prefix) - 1]
    assert ema(prefix, 3)[-1] == ema(extended, 3)[len(prefix) - 1]


@pytest.mark.parametrize(
    "function",
    [
        lambda: sma((1.0, 2.0), 0),
        lambda: ema((1.0, 2.0), 3),
        lambda: rate_of_change((1.0, 0.0, 2.0), 1),
        lambda: bollinger_bands((1.0, 2.0), 2, -1),
        lambda: vwap((1.0,), (2.0,), (1.5,), (1.0,)),
    ],
)
def test_invalid_domains(function: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        function()  # type: ignore[operator]
