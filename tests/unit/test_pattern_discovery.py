"""Tests for validated exploratory pattern records."""

import numpy as np

from core.contracts.ml import MLDataset
from ml import discover_patterns


def test_discovers_strong_association_without_claiming_causality() -> None:
    values = np.linspace(-5.0, 5.0, 80)
    dataset = MLDataset(
        dataset_id="synthetic-association-v1",
        source="synthetic test fixture",
        feature_names=("signal", "noise"),
        target_name="target",
        features=tuple((float(x), float(np.sin(x))) for x in values),
        target=tuple(float(4.0 * x + 1.0) for x in values),
        task="regression",
    )
    patterns = discover_patterns(dataset, min_abs_correlation=0.8)
    assert patterns
    assert patterns[0].feature_name == "signal"
    assert patterns[0].pearson_correlation > 0.99
    assert patterns[0].interpretation == "association_only"
    assert any("does not imply causation" in item for item in patterns[0].limitations)


def test_constant_features_do_not_create_spurious_patterns() -> None:
    dataset = MLDataset(
        dataset_id="synthetic-constant-v1",
        source="synthetic test fixture",
        feature_names=("constant",),
        target_name="target",
        features=tuple((1.0,) for _ in range(20)),
        target=tuple(float(index) for index in range(20)),
        task="regression",
    )
    assert discover_patterns(dataset) == ()
