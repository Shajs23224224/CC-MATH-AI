"""Reference and invariant tests for F15 statistical analytics."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from core.contracts import StatisticalSummary
from quant.statistics import (
    autocorrelation,
    descriptive_summary,
    excess_kurtosis,
    interquartile_range,
    median_absolute_deviation,
    quantile,
    rankdata,
    rolling_mean,
    rolling_std,
    rolling_zscore,
    skewness,
    spearman_correlation,
)


def test_descriptive_statistics() -> None:
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    summary = descriptive_summary(values)
    assert isinstance(summary, StatisticalSummary)
    assert summary.mean == pytest.approx(3.0)
    assert summary.median == pytest.approx(3.0)
    assert summary.variance == pytest.approx(2.0)
    assert summary.standard_deviation == pytest.approx(2.0**0.5)
    assert summary.minimum == 1.0
    assert summary.maximum == 5.0
    assert summary.first_quartile == pytest.approx(2.0)
    assert summary.third_quartile == pytest.approx(4.0)
    assert summary.interquartile_range == pytest.approx(2.0)
    assert summary.median_absolute_deviation == pytest.approx(1.0)


def test_robust_and_shape_statistics() -> None:
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    assert quantile(values, 0.25) == pytest.approx(2.0)
    assert interquartile_range(values) == pytest.approx(2.0)
    assert median_absolute_deviation(values) == pytest.approx(1.0)
    assert skewness(values) == pytest.approx(0.0)
    assert excess_kurtosis(values) == pytest.approx(-1.3)


def test_rank_and_dependence() -> None:
    assert rankdata((10.0, 20.0, 20.0, 30.0)) == pytest.approx((1.0, 2.5, 2.5, 4.0))
    assert spearman_correlation((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)) == pytest.approx(-1.0)
    assert autocorrelation((1.0, 2.0, 3.0, 4.0), lag=1) == pytest.approx(1.0)


def test_rolling_statistics() -> None:
    values = (1.0, 2.0, 3.0, 4.0)
    assert rolling_mean(values, 2) == pytest.approx((1.5, 2.5, 3.5))
    assert rolling_std(values, 2) == pytest.approx((0.5, 0.5, 0.5))
    assert rolling_zscore(values, 2) == pytest.approx((1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: quantile((1.0, 2.0), -0.1),
        lambda: quantile((1.0, 2.0), 1.1),
        lambda: quantile((1.0, 2.0), float("nan")),
        lambda: skewness((1.0, 1.0)),
        lambda: excess_kurtosis((1.0, 1.0)),
        lambda: spearman_correlation((1.0, 2.0), (1.0,)),
        lambda: autocorrelation((1.0, 2.0), 1),
        lambda: rolling_mean((1.0,), 2),
        lambda: rolling_std((1.0, 2.0), 1),
        lambda: rolling_zscore((1.0, 1.0), 2),
    ],
)
def test_invalid_domains_are_rejected(operation: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        operation()
