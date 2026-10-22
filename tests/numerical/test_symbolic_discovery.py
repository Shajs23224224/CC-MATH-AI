"""Numerical tests for bounded polynomial relation discovery on synthetic data."""

import numpy as np
import pytest

from core.contracts.ml import MLDataset
from ml import discover_polynomial_relation, predict_polynomial_relation
from ml.errors import MLInputError


def test_recovers_simple_affine_relation_out_of_sample() -> None:
    x = np.linspace(-4.0, 4.0, 100)
    dataset = MLDataset(
        dataset_id="synthetic-y-equals-2x-plus-3-v1",
        source="synthetic exact relation y = 2*x + 3",
        feature_names=("x",),
        target_name="y",
        features=tuple((float(value),) for value in x),
        target=tuple(float(2.0 * value + 3.0) for value in x),
        task="regression",
    )
    candidate = discover_polynomial_relation(dataset, max_degree=3, max_terms=16)
    assert candidate.degree == 1
    assert candidate.validation_rmse < 1e-8
    assert candidate.test_rmse < 1e-8
    predictions = predict_polynomial_relation(candidate, ((0.0,), (1.0,), (2.0,)))
    assert predictions == pytest.approx((3.0, 5.0, 7.0), abs=1e-8)
    assert candidate.test_used_for_selection is False
    assert candidate.status == "EXPERIMENTAL"


def test_candidate_generation_respects_grammar_budget() -> None:
    dataset = MLDataset(
        dataset_id="synthetic-budget-v1",
        source="synthetic test fixture",
        feature_names=("x1", "x2", "x3", "x4"),
        target_name="y",
        features=tuple((float(i), float(i % 2), float(i % 5), float(i % 7)) for i in range(24)),
        target=tuple(float(i * 2) for i in range(24)),
        task="regression",
    )
    with pytest.raises(MLInputError, match="max_terms"):
        discover_polynomial_relation(dataset, max_degree=4, max_terms=10)


def test_selected_candidate_can_be_reused_without_dynamic_code_execution() -> None:
    x = np.linspace(1.0, 4.0, 40)
    dataset = MLDataset(
        dataset_id="synthetic-reuse-v1",
        source="synthetic test fixture",
        feature_names=("x",),
        target_name="y",
        features=tuple((float(value),) for value in x),
        target=tuple(float(value**2) for value in x),
        task="regression",
    )
    candidate = discover_polynomial_relation(dataset, max_degree=2, max_terms=8)
    assert predict_polynomial_relation(candidate, ((2.0,),))[0] == pytest.approx(4.0, abs=1e-6)
