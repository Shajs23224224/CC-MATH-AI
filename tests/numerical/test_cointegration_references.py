"""Numerical reference and edge-case tests for F24 cointegration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from core.contracts import CointegratingPairResult, MultipleTestingResult, UnitRootTestResult
from quant.cointegration import (
    adf_test,
    benjamini_hochberg,
    engle_granger_test,
    fit_cointegrating_pair,
    johansen_test,
    kpss_test,
    phillips_perron_test,
    spread_zscore,
    unit_root_diagnostics,
)


def _stationary_ar(phi: float, innovations: np.ndarray) -> np.ndarray:
    """Build a deterministic seeded AR(1) stationary process."""
    values = np.zeros(innovations.size, dtype=np.float64)
    for index in range(1, innovations.size):
        values[index] = phi * values[index - 1] + innovations[index]
    return values


def _cointegrated_pair(size: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261022)
    common_walk = np.cumsum(rng.normal(size=size))
    stationary_spread = _stationary_ar(0.65, rng.normal(scale=0.4, size=size))
    dependent = 0.3 + 1.7 * common_walk + stationary_spread
    return dependent, common_walk


def test_adf_kpss_and_phillips_perron_return_typed_results() -> None:
    rng = np.random.default_rng(21)
    stationary = _stationary_ar(0.4, rng.normal(size=400))
    random_walk = np.cumsum(rng.normal(size=400))

    adf = adf_test(stationary, trend="c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stationary = kpss_test(stationary, trend="c")
        kpss_walk = kpss_test(random_walk, trend="c")
    pp_stationary = phillips_perron_test(stationary, trend="c")
    pp_walk = phillips_perron_test(random_walk, trend="c")

    assert isinstance(adf, UnitRootTestResult)
    assert adf.test == "adf"
    assert adf.pvalue < 0.05
    assert set(adf.critical_values) == {"1%", "5%", "10%"}
    assert 0.0 <= kpss_stationary.pvalue <= 1.0
    assert kpss_stationary.test == "kpss"
    assert kpss_walk.pvalue <= 0.10
    assert pp_stationary.test == "phillips_perron"
    assert pp_stationary.pvalue < 0.05
    assert 0.0 <= pp_walk.pvalue <= 1.0


def test_unit_root_diagnostics_preserve_distinct_null_hypotheses() -> None:
    rng = np.random.default_rng(25)
    stationary = _stationary_ar(0.3, rng.normal(size=300))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf, kpss, pp = unit_root_diagnostics(stationary, trend="c", max_lags=4)

    assert (adf.test, kpss.test, pp.test) == ("adf", "kpss", "phillips_perron")
    assert "unit root" in adf.null_hypothesis
    assert "stationary" in kpss.null_hypothesis
    assert "unit root" in pp.null_hypothesis
    assert adf.lags <= 4
    assert pp.lags <= 4


def test_engle_granger_detects_synthetic_cointegration() -> None:
    dependent, independent = _cointegrated_pair()

    result = engle_granger_test(dependent, independent, trend="c")

    assert result.test == "engle_granger"
    assert result.observations == 400
    assert result.pvalue < 0.05
    assert len(result.critical_values) == 3


def test_pair_fit_recovers_hedge_ratio_spread_and_half_life() -> None:
    dependent, independent = _cointegrated_pair()

    result = fit_cointegrating_pair(
        dependent,
        independent,
        dependent_symbol="y_asset",
        independent_symbol="x_asset",
    )

    assert isinstance(result, CointegratingPairResult)
    assert result.dependent_symbol == "Y_ASSET"
    assert result.independent_symbol == "X_ASSET"
    assert result.hedge_ratio == pytest.approx(1.7, abs=0.08)
    assert len(result.spread) == result.observations == 400
    assert result.cointegration_pvalue < 0.05
    assert result.half_life is not None
    assert result.half_life > 0.0


def test_johansen_returns_multivariate_rank_statistics() -> None:
    dependent, independent = _cointegrated_pair(300)

    result = johansen_test(
        {"Y_ASSET": dependent, "X_ASSET": independent},
        det_order=0,
        k_ar_diff=1,
    )

    assert result.assets == ("X_ASSET", "Y_ASSET")
    assert result.observations == 300
    assert len(result.eigenvalues) == 2
    assert len(result.trace_statistics) == 2
    assert len(result.trace_critical_values) == 2
    assert result.trace_rank_95 >= 1
    assert result.max_eigen_rank_95 >= 1


def test_spread_zscore_uses_trailing_windows_and_validates_inputs() -> None:
    rng = np.random.default_rng(8)
    spread = _stationary_ar(0.5, rng.normal(size=30))

    scores = spread_zscore(spread, window=10)

    assert len(scores) == 21
    assert all(np.isfinite(value) for value in scores)
    with pytest.raises(ValueError, match="window must"):
        spread_zscore(spread, window=2)
    with pytest.raises(ValueError, match="undefined for a constant"):
        spread_zscore((1.0, 1.0, 1.0, 1.0), window=3)


def test_benjamini_hochberg_controls_false_discovery_rate() -> None:
    results = benjamini_hochberg((0.001, 0.01, 0.04, 0.20), alpha=0.05)

    assert [item.original_index for item in results] == [0, 1, 2, 3]
    assert [item.rank for item in results] == [1, 2, 3, 4]
    assert [item.adjusted_pvalue for item in results] == pytest.approx(
        (0.004, 0.02, 0.05333333333333334, 0.20)
    )
    assert [item.reject_null for item in results] == [True, True, False, False]


def test_multiple_testing_contract_rejects_inconsistent_decision() -> None:
    with pytest.raises(ValidationError, match="must match"):
        MultipleTestingResult(
            original_index=0,
            rank=1,
            pvalue=0.01,
            adjusted_pvalue=0.01,
            reject_null=False,
            alpha=0.05,
        )


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ((1.0,) * 20, (1.0,) * 19, "identical lengths"),
        ((1.0,) * 19, (2.0,) * 19, "at least 20"),
        ((float("nan"),) + (1.0,) * 19, (2.0,) * 20, "finite observations"),
        ((1.0,) * 20, (2.0,) * 20, "positive sample variance"),
    ],
)
def test_engle_granger_rejects_invalid_inputs(
    left: tuple[float, ...],
    right: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        engle_granger_test(left, right)


def test_cointegration_helpers_reject_invalid_lags_and_series_labels() -> None:
    dependent, independent = _cointegrated_pair(100)
    with pytest.raises(ValueError, match="max_lags"):
        adf_test(dependent, max_lags=-1)
    with pytest.raises(ValueError, match="must be distinct"):
        fit_cointegrating_pair(
            dependent,
            independent,
            dependent_symbol="ABC",
            independent_symbol="abc",
        )
    with pytest.raises(ValueError, match="duplicate series symbol"):
        johansen_test({"ABC": dependent, " abc ": independent})
    with pytest.raises(ValueError, match="at least one p-value"):
        benjamini_hochberg(())


def test_unit_root_contract_requires_standard_critical_value_levels() -> None:
    with pytest.raises(ValidationError, match="must include 1%, 5% and 10%"):
        UnitRootTestResult(
            test="adf",
            statistic=-4.0,
            pvalue=0.001,
            lags=2,
            observations=100,
            trend="c",
            critical_values={"5%": -2.8, "10%": -2.5},
            null_hypothesis="unit root",
            alternative_hypothesis="stationary",
        )
