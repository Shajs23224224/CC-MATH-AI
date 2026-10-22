"""Numerical and edge-case tests for F23 cross-asset dependency."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from core.contracts import DependenceMatrix
from quant.dependence import (
    correlation_matrix,
    covariance_matrix,
    oas_shrinkage_covariance,
    pca_factors,
    rolling_dependency,
)


def _returns_panel(size: int = 60) -> dict[str, tuple[float, ...]]:
    rng = np.random.default_rng(1234)
    common = rng.normal(0.0005, 0.02, size=size)
    independent = rng.normal(-0.0001, 0.01, size=size)
    return {
        "zzz": tuple((0.8 * common + 0.2 * independent).tolist()),
        "aaa": tuple(common.tolist()),
        "mmm": tuple((-0.4 * common + 0.7 * independent).tolist()),
    }


def test_pearson_matrix_is_sorted_symmetric_bounded_and_psd() -> None:
    matrix = correlation_matrix(_returns_panel())

    values = np.asarray(matrix.matrix)
    assert matrix.assets == ("AAA", "MMM", "ZZZ")
    assert matrix.method == "pearson"
    assert matrix.observations == 60
    assert matrix.start_index == 0
    assert matrix.end_index == 60
    assert np.allclose(values, values.T)
    assert np.allclose(np.diag(values), 1.0)
    assert np.min(np.linalg.eigvalsh(values)) >= -1e-8
    assert np.all(np.abs(values) <= 1.0 + 1e-8)


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_rank_correlation_handles_monotonic_dependence(method: str) -> None:
    left = tuple(float(value) for value in range(1, 31))
    panel = {"B": tuple(-value for value in left), "A": left}

    matrix = correlation_matrix(panel, method=method)

    assert matrix.assets == ("A", "B")
    assert matrix.matrix[0][1] == pytest.approx(-1.0)
    assert matrix.matrix[1][0] == pytest.approx(-1.0)


def test_covariance_matches_sample_covariance_reference() -> None:
    panel = _returns_panel(20)
    result = covariance_matrix(panel)
    reference = np.cov(
        np.asarray([panel[name] for name in result.assets], dtype=float),
        ddof=1,
    )

    assert result.method == "sample_covariance"
    assert result.kind == "covariance"
    assert np.allclose(np.asarray(result.matrix), reference)


def test_oas_shrinkage_is_reproducible_and_psd() -> None:
    panel = _returns_panel()
    first = oas_shrinkage_covariance(panel)
    second = oas_shrinkage_covariance(panel)
    values = np.asarray(first.matrix)

    assert first == second
    assert first.method == "oas_shrinkage"
    assert first.shrinkage_intensity is not None
    assert 0.0 <= first.shrinkage_intensity <= 1.0
    assert np.min(np.linalg.eigvalsh(values)) >= -1e-8
    assert np.allclose(values, values.T)


def test_rolling_dependency_has_end_exclusive_window_indices() -> None:
    series = tuple(float(value) / 100 for value in range(80))
    panel = {
        "B": series,
        "A": tuple(value * 1.5 + 0.001 * (index % 3) for index, value in enumerate(series)),
    }

    rolling = rolling_dependency(panel, window=20)

    assert len(rolling) == 61
    assert (rolling[0].start_index, rolling[0].end_index) == (0, 20)
    assert (rolling[-1].start_index, rolling[-1].end_index) == (60, 80)
    assert all(item.observations == 20 for item in rolling)
    assert all(item.assets == ("A", "B") for item in rolling)


def test_pca_factors_are_ordered_orthonormal_and_repeatable() -> None:
    first = pca_factors(_returns_panel(), n_components=2)
    second = pca_factors(_returns_panel(), n_components=2)
    loadings = np.asarray(first.loadings)
    scores = np.asarray(first.scores)

    assert first == second
    assert first.assets == ("AAA", "MMM", "ZZZ")
    assert len(first.eigenvalues) == 2
    assert first.eigenvalues[0] >= first.eigenvalues[1] >= 0.0
    assert sum(first.explained_variance_ratio) <= 1.0 + 1e-8
    assert loadings.shape == (3, 2)
    assert np.allclose(loadings.T @ loadings, np.eye(2), atol=1e-7)
    assert scores.shape == (60, 2)


def test_standardized_pca_rejects_constant_series() -> None:
    panel = {
        "A": tuple(float(value) for value in range(10)),
        "B": (1.0,) * 10,
    }
    with pytest.raises(ValueError, match="positive finite variance"):
        pca_factors(panel, standardize=True)


@pytest.mark.parametrize(
    ("panel", "message"),
    [
        ({"A": (0.1, 0.2, 0.3), "B": (0.1, 0.2)}, "same length"),
        ({" A ": (0.1, 0.2, 0.3), "a": (0.2, 0.1, 0.4)}, "duplicate asset"),
        ({"A": (0.1, float("nan"), 0.3), "B": (0.3, 0.2, 0.1)}, "finite"),
        ({"A": (0.1, 0.1, 0.1), "B": (0.2, 0.3, 0.4)}, "constant"),
    ],
)
def test_invalid_dependence_inputs_fail_explicitly(
    panel: dict[str, tuple[float, ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        correlation_matrix(panel)


def test_dependence_contract_rejects_indefinite_correlation_matrix() -> None:
    with pytest.raises(ValidationError, match="positive semidefinite"):
        DependenceMatrix(
            assets=("A", "B", "C"),
            method="pearson",
            kind="correlation",
            observations=10,
            start_index=0,
            end_index=10,
            matrix=(
                (1.0, 0.9, 0.9),
                (0.9, 1.0, -0.9),
                (0.9, -0.9, 1.0),
            ),
        )


def test_rolling_window_and_component_count_are_validated() -> None:
    panel = _returns_panel(30)
    with pytest.raises(ValueError, match="window must"):
        rolling_dependency(panel, window=2)
    with pytest.raises(ValueError, match="cannot exceed"):
        rolling_dependency(panel, window=31)
    with pytest.raises(ValueError, match="n_components"):
        pca_factors(panel, n_components=4)
