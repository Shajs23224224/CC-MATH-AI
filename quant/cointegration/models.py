"""Unit-root tests, cointegration and pairs statistics for F24."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from arch.unitroot import PhillipsPerron  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import adfuller, coint, kpss  # type: ignore[import-untyped]
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore[import-untyped]

from core.contracts import (
    CointegratingPairResult,
    CointegrationTestResult,
    JohansenTestResult,
    MultipleTestingResult,
    UnitRootTestResult,
)

_MIN_OBSERVATIONS = 20
_EPSILON = 1e-12
_UnitRootTrend = Literal["c", "ct"]
_CointegrationTrend = Literal["c", "ct", "ctt", "n"]


def _validate_series(
    values: Sequence[float],
    *,
    name: str,
    minimum: int = _MIN_OBSERVATIONS,
    require_variance: bool = False,
) -> np.ndarray:
    """Return a finite one-dimensional numeric sample."""
    try:
        checked = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from exc
    if checked.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if checked.size < minimum:
        raise ValueError(f"{name} requires at least {minimum} observations")
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain only finite observations; missing data is not imputed")
    if require_variance and float(np.var(checked, ddof=1)) <= _EPSILON:
        raise ValueError(f"{name} must have positive sample variance")
    return checked


def _validate_max_lags(max_lags: int | None, observations: int) -> int | None:
    """Validate optional lag selection for unit-root and cointegration tests."""
    if max_lags is None:
        return None
    if not isinstance(max_lags, int) or isinstance(max_lags, bool) or max_lags < 0:
        raise ValueError("max_lags must be a non-negative integer or None")
    if max_lags >= observations // 2:
        raise ValueError("max_lags must be smaller than half the sample size")
    return max_lags


def _critical_values(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize significance labels and values from a test-library result."""
    required = ("1%", "5%", "10%")
    result = {key: float(values[key]) for key in required if key in values}
    if len(result) != 3 or not all(math.isfinite(value) for value in result.values()):
        raise ValueError("test returned invalid or incomplete critical values")
    return result


def adf_test(
    values: Sequence[float],
    *,
    trend: _UnitRootTrend = "c",
    max_lags: int | None = None,
) -> UnitRootTestResult:
    """Run the Augmented Dickey-Fuller unit-root test.

    Null: a unit root is present. Alternative: stationarity around the selected
    constant or constant-plus-linear-trend deterministic component.
    """
    checked = _validate_series(values, name="values", require_variance=True)
    lag_limit = _validate_max_lags(max_lags, int(checked.size))
    result = adfuller(
        checked,
        maxlag=lag_limit,
        regression=trend,
        autolag="AIC" if lag_limit is None else None,
    )
    statistic, pvalue, used_lags, observations, critical, *_ = result
    return UnitRootTestResult(
        test="adf",
        statistic=float(statistic),
        pvalue=float(pvalue),
        lags=int(used_lags),
        observations=int(observations),
        trend=trend,
        critical_values=_critical_values(critical),
        null_hypothesis="series contains a unit root",
        alternative_hypothesis="series is stationary around the selected deterministic trend",
    )


def kpss_test(
    values: Sequence[float],
    *,
    trend: _UnitRootTrend = "c",
) -> UnitRootTestResult:
    """Run the KPSS test, whose null is stationarity."""
    checked = _validate_series(values, name="values", require_variance=True)
    statistic, pvalue, used_lags, critical = kpss(checked, regression=trend, nlags="auto")
    return UnitRootTestResult(
        test="kpss",
        statistic=float(statistic),
        pvalue=float(pvalue),
        lags=int(used_lags),
        observations=int(checked.size),
        trend=trend,
        critical_values=_critical_values(critical),
        null_hypothesis=(
            "series is level-stationary" if trend == "c" else "series is trend-stationary"
        ),
        alternative_hypothesis="series contains a unit root or is not stationary",
    )


def phillips_perron_test(
    values: Sequence[float],
    *,
    trend: _UnitRootTrend = "c",
    max_lags: int | None = None,
) -> UnitRootTestResult:
    """Run the Phillips-Perron unit-root test using Newey-West long-run variance."""
    checked = _validate_series(values, name="values", require_variance=True)
    lag_limit = _validate_max_lags(max_lags, int(checked.size))
    result = PhillipsPerron(
        checked,
        lags=lag_limit,
        trend=trend,
        test_type="tau",
    )
    critical = _critical_values(result.critical_values)
    return UnitRootTestResult(
        test="phillips_perron",
        statistic=float(result.stat),
        pvalue=float(result.pvalue),
        lags=int(result.lags),
        observations=int(checked.size),
        trend=trend,
        critical_values=critical,
        null_hypothesis="series contains a unit root",
        alternative_hypothesis="series is stationary after Phillips-Perron correction",
    )


def unit_root_diagnostics(
    values: Sequence[float],
    *,
    trend: _UnitRootTrend = "c",
    max_lags: int | None = None,
) -> tuple[UnitRootTestResult, UnitRootTestResult, UnitRootTestResult]:
    """Run ADF, KPSS and Phillips-Perron, retaining their different null hypotheses."""
    checked = tuple(float(value) for value in values)
    return (
        adf_test(checked, trend=trend, max_lags=max_lags),
        kpss_test(checked, trend=trend),
        phillips_perron_test(checked, trend=trend, max_lags=max_lags),
    )


def _validate_pair(
    dependent: Sequence[float],
    independent: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a temporally aligned series pair."""
    if len(dependent) != len(independent):
        raise ValueError("dependent and independent series must have identical lengths")
    left = _validate_series(dependent, name="dependent", require_variance=True)
    right = _validate_series(independent, name="independent", require_variance=True)
    return left, right


def engle_granger_test(
    dependent: Sequence[float],
    independent: Sequence[float],
    *,
    trend: _CointegrationTrend = "c",
    max_lags: int | None = None,
) -> CointegrationTestResult:
    """Run the Engle-Granger two-step test of the null of no cointegration."""
    left, right = _validate_pair(dependent, independent)
    lag_limit = _validate_max_lags(max_lags, int(left.size))
    statistic, pvalue, critical = coint(
        left,
        right,
        trend=trend,
        maxlag=lag_limit,
        autolag="aic" if lag_limit is None else None,
    )
    critical_values = tuple(float(value) for value in critical)
    if not math.isfinite(float(statistic)) or not math.isfinite(float(pvalue)):
        raise ValueError("Engle-Granger test returned non-finite results; check for collinearity")
    if not all(math.isfinite(value) for value in critical_values):
        raise ValueError("Engle-Granger critical values must be finite")
    return CointegrationTestResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        observations=int(left.size),
        trend=trend,
        critical_values=(critical_values[0], critical_values[1], critical_values[2]),
    )


def _prepare_multivariate(
    series: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate and sort a complete multivariate level panel."""
    if len(series) < 2:
        raise ValueError("at least two series are required for Johansen cointegration")
    normalized: dict[str, tuple[float, ...]] = {}
    length: int | None = None
    for symbol, values in series.items():
        if not isinstance(symbol, str):
            raise ValueError("series symbols must be strings")
        name = symbol.strip().upper()
        if not name:
            raise ValueError("series symbols must be non-empty")
        if name in normalized:
            raise ValueError(f"duplicate series symbol after normalization: {name}")
        checked = _validate_series(values, name=name)
        if length is None:
            length = int(checked.size)
        elif checked.size != length:
            raise ValueError("all series must have identical lengths; implicit alignment is forbidden")
        if float(np.var(checked, ddof=1)) <= _EPSILON:
            raise ValueError(f"series {name} must have positive sample variance")
        normalized[name] = tuple(float(value) for value in checked)

    assets = tuple(sorted(normalized))
    panel = np.asarray([normalized[asset] for asset in assets], dtype=np.float64).T
    return assets, panel


def johansen_test(
    series: Mapping[str, Sequence[float]],
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenTestResult:
    """Run the multivariate Johansen trace and max-eigenvalue tests."""
    assets, panel = _prepare_multivariate(series)
    observations, asset_count = panel.shape
    if not isinstance(det_order, int) or isinstance(det_order, bool) or det_order not in {-1, 0, 1}:
        raise ValueError("det_order must be -1, 0 or 1")
    if not isinstance(k_ar_diff, int) or isinstance(k_ar_diff, bool) or not 0 <= k_ar_diff < observations - 1:
        raise ValueError("k_ar_diff must be a non-negative lag order below observations - 1")

    result = coint_johansen(panel, det_order=det_order, k_ar_diff=k_ar_diff)
    trace_critical = np.asarray(result.trace_stat_crit_vals, dtype=np.float64)
    max_critical = np.asarray(result.max_eig_stat_crit_vals, dtype=np.float64)
    trace_statistics = np.asarray(result.trace_stat, dtype=np.float64)
    max_statistics = np.asarray(result.max_eig_stat, dtype=np.float64)

    def selected_rank(statistics: np.ndarray, critical_values: np.ndarray) -> int:
        """Use sequential 5% trace/max-eigen tests to estimate cointegration rank."""
        for index in range(asset_count):
            if statistics[index] <= critical_values[index, 1]:
                return index
        return asset_count

    return JohansenTestResult(
        assets=assets,
        observations=observations,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        eigenvalues=tuple(float(value) for value in result.eig),
        trace_statistics=tuple(float(value) for value in trace_statistics),
        max_eigen_statistics=tuple(float(value) for value in max_statistics),
        trace_critical_values=tuple(
            tuple(float(value) for value in row) for row in trace_critical
        ),
        max_eigen_critical_values=tuple(
            tuple(float(value) for value in row) for row in max_critical
        ),
        trace_rank_95=selected_rank(trace_statistics, trace_critical),
        max_eigen_rank_95=selected_rank(max_statistics, max_critical),
    )


def _estimate_half_life(spread: np.ndarray) -> float | None:
    """Estimate mean-reversion half-life from a first-order spread dynamics model."""
    lagged = spread[:-1]
    differences = np.diff(spread)
    design = np.column_stack((np.ones(lagged.size, dtype=np.float64), lagged))
    coefficients, *_ = np.linalg.lstsq(design, differences, rcond=None)
    rho = 1.0 + float(coefficients[1])
    if not math.isfinite(rho) or not 0.0 < rho < 1.0:
        return None
    half_life = -math.log(2.0) / math.log(rho)
    if not math.isfinite(half_life) or half_life <= 0.0:
        return None
    return float(half_life)


def fit_cointegrating_pair(
    dependent: Sequence[float],
    independent: Sequence[float],
    *,
    dependent_symbol: str,
    independent_symbol: str,
    max_lags: int | None = None,
) -> CointegratingPairResult:
    """Estimate OLS hedge ratio, spread, Engle-Granger p-value and half-life.

    The fitted intercept and hedge ratio are in-sample parameters. In a backtest,
    estimate them only on the training window, then hold them fixed in the
    subsequent test/forward window to avoid look-ahead leakage.
    """
    if not dependent_symbol.strip() or not independent_symbol.strip():
        raise ValueError("pair symbols must be non-empty")
    if dependent_symbol.strip().upper() == independent_symbol.strip().upper():
        raise ValueError("pair symbols must be distinct")
    left, right = _validate_pair(dependent, independent)
    design = np.column_stack((np.ones(right.size, dtype=np.float64), right))
    coefficients, *_ = np.linalg.lstsq(design, left, rcond=None)
    intercept, hedge_ratio = float(coefficients[0]), float(coefficients[1])
    spread = left - (intercept + hedge_ratio * right)
    test = engle_granger_test(left, right, trend="c", max_lags=max_lags)
    return CointegratingPairResult(
        dependent_symbol=dependent_symbol.strip().upper(),
        independent_symbol=independent_symbol.strip().upper(),
        observations=int(left.size),
        intercept=intercept,
        hedge_ratio=hedge_ratio,
        spread=tuple(float(value) for value in spread),
        cointegration_statistic=test.statistic,
        cointegration_pvalue=test.pvalue,
        critical_values=test.critical_values,
        half_life=_estimate_half_life(spread),
    )


def spread_zscore(
    spread: Sequence[float],
    *,
    window: int,
) -> tuple[float, ...]:
    """Calculate trailing z-scores using each window including its current sample."""
    checked = _validate_series(spread, name="spread", minimum=3)
    if not isinstance(window, int) or isinstance(window, bool) or window < 3:
        raise ValueError("window must be an integer of at least 3")
    if window > checked.size:
        raise ValueError("window cannot exceed spread length")

    result: list[float] = []
    for end in range(window, checked.size + 1):
        sample = checked[end - window : end]
        scale = float(np.std(sample, ddof=1))
        if not math.isfinite(scale) or scale <= _EPSILON:
            raise ValueError("spread z-score is undefined for a constant window")
        result.append(float((sample[-1] - np.mean(sample)) / scale))
    return tuple(result)


def benjamini_hochberg(
    pvalues: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[MultipleTestingResult, ...]:
    """Apply Benjamini-Hochberg false-discovery-rate control.

    Results preserve the input order and include each test's sorted rank,
    adjusted p-value and decision for the supplied family-wise hypothesis set.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if len(pvalues) == 0:
        raise ValueError("at least one p-value is required")
    checked = tuple(float(value) for value in pvalues)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in checked):
        raise ValueError("p-values must be finite and between 0 and 1")

    ordered = sorted(enumerate(checked), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[int, tuple[int, float]] = {}
    running_min = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index, value = ordered[rank_index]
        rank = rank_index + 1
        running_min = min(running_min, value * count / rank, 1.0)
        adjusted[original_index] = (rank, running_min)

    return tuple(
        MultipleTestingResult(
            original_index=index,
            rank=adjusted[index][0],
            pvalue=value,
            adjusted_pvalue=adjusted[index][1],
            reject_null=adjusted[index][1] <= alpha,
            alpha=alpha,
        )
        for index, value in enumerate(checked)
    )
