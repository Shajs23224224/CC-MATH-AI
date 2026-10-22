"""Cross-asset dependence, covariance shrinkage and PCA for F23."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from scipy.stats import kendalltau  # type: ignore[import-untyped]

from core.contracts import DependenceMatrix, PCAResult
from quant.statistics import rankdata

CorrelationMethod = Literal["pearson", "spearman", "kendall"]
_MIN_OBSERVATIONS = 3
_EPSILON = 1e-12


def _prepare_panel(
    returns: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate a complete panel and return canonical sorted asset columns."""
    if len(returns) < 2:
        raise ValueError("at least two asset return series are required")
    normalized: dict[str, tuple[float, ...]] = {}
    expected_length: int | None = None
    for symbol, series in returns.items():
        if not isinstance(symbol, str):
            raise ValueError("asset symbols must be strings")
        asset = symbol.strip().upper()
        if not asset:
            raise ValueError("asset symbols must be non-empty")
        if asset in normalized:
            raise ValueError(f"duplicate asset symbol after normalization: {asset}")
        try:
            values = tuple(float(value) for value in series)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"returns for {asset} must be numeric") from exc
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("all asset return series must have the same length; no implicit alignment")
        if len(values) < _MIN_OBSERVATIONS:
            raise ValueError(f"at least {_MIN_OBSERVATIONS} observations are required")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"returns for {asset} must all be finite; missing data is not imputed")
        normalized[asset] = values

    assets = tuple(sorted(normalized))
    panel = np.asarray([normalized[asset] for asset in assets], dtype=np.float64).T
    if panel.ndim != 2 or panel.shape[1] < 2:
        raise ValueError("returns panel must be two-dimensional with at least two assets")
    return assets, panel


def _correlation_values(panel: np.ndarray, method: CorrelationMethod) -> np.ndarray:
    """Calculate a complete-sample correlation matrix with explicit tie handling."""
    if panel.shape[0] < _MIN_OBSERVATIONS:
        raise ValueError(f"at least {_MIN_OBSERVATIONS} observations are required")
    if not np.all(np.isfinite(panel)):
        raise ValueError("returns panel must be finite")
    if np.any(np.std(panel, axis=0, ddof=1) <= _EPSILON):
        raise ValueError("correlation is undefined for a constant return series")

    if method == "pearson":
        matrix = np.corrcoef(panel, rowvar=False)
    elif method == "spearman":
        ranked = np.column_stack([rankdata(panel[:, index]) for index in range(panel.shape[1])])
        matrix = np.corrcoef(ranked, rowvar=False)
    else:
        asset_count = panel.shape[1]
        matrix = np.eye(asset_count, dtype=np.float64)
        for left in range(asset_count):
            for right in range(left + 1, asset_count):
                statistic = kendalltau(panel[:, left], panel[:, right], variant="b").statistic
                if statistic is None or not math.isfinite(float(statistic)):
                    raise ValueError("Kendall correlation is undefined for the supplied series")
                matrix[left, right] = float(statistic)
                matrix[right, left] = float(statistic)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation calculation produced non-finite values")
    return np.asarray(matrix, dtype=np.float64)


def correlation_matrix(
    returns: Mapping[str, Sequence[float]],
    *,
    method: CorrelationMethod = "pearson",
) -> DependenceMatrix:
    """Return Pearson, Spearman or Kendall tau-b correlation for a complete panel.

    Inputs are period returns, not price levels. All series must share exactly
    the same timestamps and contain finite values; missing data is rejected
    rather than dropped or imputed. Asset labels are normalized and sorted.
    """
    assets, panel = _prepare_panel(returns)
    matrix = _correlation_values(panel, method)
    observations = int(panel.shape[0])
    return DependenceMatrix(
        assets=assets,
        method=method,
        kind="correlation",
        observations=observations,
        start_index=0,
        end_index=observations,
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
    )


def covariance_matrix(returns: Mapping[str, Sequence[float]]) -> DependenceMatrix:
    """Return unbiased sample covariance (ddof=1) for a complete return panel."""
    assets, panel = _prepare_panel(returns)
    matrix = np.asarray(np.cov(panel, rowvar=False, ddof=1), dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance calculation produced non-finite values")
    observations = int(panel.shape[0])
    return DependenceMatrix(
        assets=assets,
        method="sample_covariance",
        kind="covariance",
        observations=observations,
        start_index=0,
        end_index=observations,
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
    )


def oas_shrinkage_covariance(returns: Mapping[str, Sequence[float]]) -> DependenceMatrix:
    """Return an Oracle Approximating Shrinkage covariance estimate.

    The empirical maximum-likelihood covariance (divisor n) is shrunk toward
    a scaled identity target. The intensity is estimated analytically and is
    returned with the matrix so the regularization is auditable.
    """
    assets, panel = _prepare_panel(returns)
    centered = panel - np.mean(panel, axis=0, keepdims=True)
    observations, asset_count = centered.shape
    empirical = centered.T @ centered / observations
    mean_variance = float(np.trace(empirical) / asset_count)
    alpha = float(np.mean(empirical * empirical))
    denominator = (observations + 1.0) * (alpha - mean_variance**2 / asset_count)
    if not math.isfinite(denominator) or denominator <= _EPSILON:
        shrinkage = 1.0
    else:
        shrinkage = min(max((alpha + mean_variance**2) / denominator, 0.0), 1.0)
    target = np.eye(asset_count, dtype=np.float64) * mean_variance
    matrix = (1.0 - shrinkage) * empirical + shrinkage * target
    matrix = (matrix + matrix.T) / 2.0
    if not np.all(np.isfinite(matrix)):
        raise ValueError("OAS shrinkage covariance produced non-finite values")
    return DependenceMatrix(
        assets=assets,
        method="oas_shrinkage",
        kind="covariance",
        observations=int(observations),
        start_index=0,
        end_index=int(observations),
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        shrinkage_intensity=float(shrinkage),
    )


def rolling_dependency(
    returns: Mapping[str, Sequence[float]],
    *,
    window: int,
    method: CorrelationMethod = "pearson",
) -> tuple[DependenceMatrix, ...]:
    """Return rolling correlation matrices with explicit end-exclusive indices."""
    assets, panel = _prepare_panel(returns)
    if not isinstance(window, int) or isinstance(window, bool) or window < _MIN_OBSERVATIONS:
        raise ValueError(f"window must be an integer of at least {_MIN_OBSERVATIONS}")
    if window > panel.shape[0]:
        raise ValueError("window cannot exceed the number of observations")
    result: list[DependenceMatrix] = []
    for end_index in range(window, panel.shape[0] + 1):
        start_index = end_index - window
        matrix = _correlation_values(panel[start_index:end_index], method)
        result.append(
            DependenceMatrix(
                assets=assets,
                method=method,
                kind="correlation",
                observations=window,
                start_index=start_index,
                end_index=end_index,
                matrix=tuple(tuple(float(value) for value in row) for row in matrix),
            )
        )
    return tuple(result)


def pca_factors(
    returns: Mapping[str, Sequence[float]],
    *,
    n_components: int | None = None,
    standardize: bool = False,
) -> PCAResult:
    """Fit deterministic PCA factors to centered return observations.

    Components are ordered by descending eigenvalue. Each loading vector's
    sign is fixed so its largest absolute loading is non-negative, providing a
    stable representation across repeated calls (up to degenerate eigenspaces).
    Set standardize=True to perform PCA on standardized series rather than
    letting higher-variance assets dominate the covariance matrix.
    """
    assets, panel = _prepare_panel(returns)
    asset_count = panel.shape[1]
    if n_components is None:
        components = asset_count
    elif (
        not isinstance(n_components, int)
        or isinstance(n_components, bool)
        or not 1 <= n_components <= asset_count
    ):
        raise ValueError("n_components must be a positive integer not exceeding the asset count")
    else:
        components = n_components

    centered = panel - np.mean(panel, axis=0, keepdims=True)
    if standardize:
        scales = np.std(centered, axis=0, ddof=1)
        if np.any(scales <= _EPSILON) or not np.all(np.isfinite(scales)):
            raise ValueError("standardized PCA requires positive finite variance for every asset")
        transformed = centered / scales
    else:
        transformed = centered

    covariance = transformed.T @ transformed / (transformed.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    all_eigenvalues = np.maximum(eigenvalues[order], 0.0)
    all_eigenvectors = eigenvectors[:, order]
    total_variance = float(np.sum(all_eigenvalues))
    if not math.isfinite(total_variance) or total_variance <= _EPSILON:
        raise ValueError("PCA requires positive total variance")

    loadings = all_eigenvectors[:, :components].copy()
    for component in range(components):
        pivot = int(np.argmax(np.abs(loadings[:, component])))
        if loadings[pivot, component] < 0.0:
            loadings[:, component] *= -1.0
    scores = transformed @ loadings
    explained = all_eigenvalues[:components] / total_variance
    return PCAResult(
        assets=assets,
        observations=int(panel.shape[0]),
        standardized=standardize,
        eigenvalues=tuple(float(value) for value in all_eigenvalues[:components]),
        explained_variance_ratio=tuple(float(value) for value in explained),
        loadings=tuple(tuple(float(value) for value in row) for row in loadings),
        scores=tuple(tuple(float(value) for value in row) for row in scores),
    )
