"""Reference and invariant tests for F13 mathematical primitives."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from core.contracts import OptimizationResult
from quant.math import (
    annuity_future_value,
    annuity_payment,
    annuity_present_value,
    arithmetic_mean,
    binomial_pmf,
    compound_growth,
    correlation,
    discount_factor,
    dot_product,
    future_value,
    geometric_mean,
    golden_section_minimize,
    harmonic_mean,
    linear_interpolate,
    linear_interpolate_series,
    normal_cdf,
    normal_pdf,
    population_std,
    population_variance,
    present_value,
    sample_covariance,
    sample_std,
    sample_variance,
    weighted_mean,
)


def test_means_match_reference_values() -> None:
    values = (1.0, 2.0, 4.0)
    assert arithmetic_mean(values) == pytest.approx(7.0 / 3.0)
    assert weighted_mean(values, (1.0, 2.0, 1.0)) == pytest.approx(2.25)
    assert geometric_mean(values) == pytest.approx(2.0)
    assert harmonic_mean(values) == pytest.approx(12.0 / 7.0)


def test_dot_product_and_compounding() -> None:
    assert dot_product((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == pytest.approx(32.0)
    assert compound_growth(1_000.0, 0.05, 2) == pytest.approx(1_102.5)


def test_variance_covariance_and_correlation() -> None:
    left = (1.0, 2.0, 3.0, 4.0)
    right = (2.0, 4.0, 6.0, 8.0)

    assert population_variance(left) == pytest.approx(1.25)
    assert sample_variance(left) == pytest.approx(5.0 / 3.0)
    assert population_std(left) == pytest.approx(math.sqrt(1.25))
    assert sample_std(left) == pytest.approx(math.sqrt(5.0 / 3.0))
    assert sample_covariance(left, right) == pytest.approx(10.0 / 3.0)
    assert correlation(left, right) == pytest.approx(1.0)


def test_probability_reference_values() -> None:
    assert normal_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert binomial_pmf(10, 3, 0.5) == pytest.approx(0.1171875)


def test_time_value_of_money_reference_values() -> None:
    assert discount_factor(0.10, 2) == pytest.approx(1.0 / 1.21)
    assert present_value(1_210.0, 0.10, 2) == pytest.approx(1_000.0)
    assert future_value(1_000.0, 0.10, 2) == pytest.approx(1_210.0)
    assert annuity_present_value(100.0, 0.10, 3) == pytest.approx(248.685199098)
    assert annuity_future_value(100.0, 0.10, 3) == pytest.approx(331.0)
    assert annuity_payment(248.685199098, 0.10, 3) == pytest.approx(100.0, abs=1e-9)


def test_zero_rate_annuity_identity() -> None:
    assert annuity_present_value(100.0, 0.0, 12) == pytest.approx(1_200.0)
    assert annuity_future_value(100.0, 0.0, 12) == pytest.approx(1_200.0)
    assert annuity_payment(1_200.0, 0.0, 12) == pytest.approx(100.0)


def test_interpolation_reference_values() -> None:
    assert linear_interpolate(0.0, 0.0, 10.0, 20.0, 2.5) == pytest.approx(5.0)
    assert linear_interpolate(0.0, 0.0, 10.0, 20.0, 15.0) == pytest.approx(30.0)
    assert linear_interpolate_series(
        (0.0, 10.0, 20.0),
        (0.0, 100.0, 50.0),
        15.0,
    ) == pytest.approx(75.0)


def test_golden_section_finds_known_quadratic_minimum() -> None:
    result = golden_section_minimize(lambda x: (x - 3.0) ** 2 + 2.0, 0.0, 10.0)
    assert isinstance(result, OptimizationResult)
    assert result.converged is True
    assert result.optimum == pytest.approx(3.0, abs=1e-7)
    assert result.objective_value == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: arithmetic_mean(()),
        lambda: weighted_mean((1.0, 2.0), (1.0,)),
        lambda: geometric_mean((1.0, 0.0)),
        lambda: harmonic_mean((1.0, 0.0)),
        lambda: sample_variance((1.0,)),
        lambda: sample_covariance((1.0,), (1.0,)),
        lambda: correlation((1.0, 1.0), (2.0, 3.0)),
        lambda: normal_pdf(0.0, std=0.0),
        lambda: normal_cdf(0.0, std=0.0),
        lambda: binomial_pmf(10, 11, 0.5),
        lambda: discount_factor(-1.0, 1),
        lambda: linear_interpolate(0.0, 0.0, 0.0, 1.0, 0.5),
        lambda: golden_section_minimize(lambda x: x, 1.0, 0.0),
    ],
)
def test_invalid_domains_are_rejected(operation: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        operation()
