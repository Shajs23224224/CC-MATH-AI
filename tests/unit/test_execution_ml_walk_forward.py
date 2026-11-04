"""Tests for three-class execution-aware ML evaluation."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from core.contracts.ml import MLDataset
from scripts.run_execution_ml_walk_forward import (
    evaluate_execution_fold,
    load_execution_dataset,
)
from scripts.run_ml_walk_forward import build_expanding_folds


def create_execution_file(tmp_path, size=120):
    dates = pd.bdate_range("2020-01-02", periods=size)

    labels = np.array(
        ["DOWN", "NEUTRAL", "UP"] * (size // 3)
        + ["DOWN", "NEUTRAL", "UP"][: size % 3]
    )

    frame = pd.DataFrame(
        {
            "feature_a": np.arange(size, dtype=float),
            "feature_b": np.arange(size, dtype=float) % 3,
            "entry_date": dates.strftime("%Y-%m-%d"),
            "entry_open": np.full(size, 100.0),
            "exit_close": np.full(size, 100.1),
            "target_execution_return": np.full(size, 0.001),
            "target_execution_direction": labels,
        },
        index=dates,
    )
    frame.index.name = "feature_date"

    dataset_path = tmp_path / "execution_dataset.csv"
    frame.to_csv(dataset_path, index_label="feature_date", float_format="%.17g")

    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    manifest = {
        "experiment": {
            "ticker": "TEST",
            "threshold_bps": 10.0,
        },
        "artifact": {
            "file": dataset_path.name,
            "sha256": digest,
        },
        "dataset": {
            "rows": size,
            "feature_names": ["feature_a", "feature_b"],
        },
        "target": {
            "classification_name": "target_execution_direction",
        },
    }

    (tmp_path / "execution_dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return dataset_path


def classification_dataset(size=120):
    features = tuple(
        (float(index), float(index % 3))
        for index in range(size)
    )
    labels = tuple(
        ("DOWN", "NEUTRAL", "UP")[index % 3]
        for index in range(size)
    )

    return MLDataset(
        dataset_id="synthetic-three-class",
        source="synthetic test data",
        feature_names=("trend", "cycle"),
        target_name="direction",
        features=features,
        target=labels,
        task="classification",
    )


def test_loader_excludes_execution_prices_from_features(tmp_path):
    path = create_execution_file(tmp_path)

    dataset, dates, manifest = load_execution_dataset(path)

    assert dataset.task == "classification"
    assert dataset.feature_names == ("feature_a", "feature_b")
    assert set(dataset.target) == {"DOWN", "NEUTRAL", "UP"}
    assert len(dates) == 120
    assert manifest["experiment"]["threshold_bps"] == 10.0


def test_loader_rejects_tampered_dataset(tmp_path):
    path = create_execution_file(tmp_path)

    with path.open("a", encoding="utf-8") as file:
        file.write("\n")

    with pytest.raises(ValueError, match="SHA-256"):
        load_execution_dataset(path)


def test_three_class_walk_forward_probabilities_and_metrics():
    dataset = classification_dataset()
    fold = build_expanding_folds(120)[0]
    dates = tuple(f"row-{index}" for index in range(120))

    record, predictions = evaluate_execution_fold(
        dataset,
        fold,
        algorithm="logistic_regression",
        seed=28,
        dates=dates,
    )

    assert record["status"] == "COMPLETED"
    assert record["training_samples"] == 72
    assert record["test_samples"] == 12
    assert record["test_used_for_selection"] is False

    assert 0.0 <= record["multiclass_log_loss"] < float("inf")
    assert 0.0 <= record["multiclass_brier_score"] <= 2.0

    assert len(predictions) == 12

    for row in predictions:
        keys = [
            "probability_down",
            "probability_neutral",
            "probability_up",
        ]
        assert all(key in row for key in keys)
        assert sum(row[key] for key in keys) == pytest.approx(1.0)


def test_three_class_evaluation_requires_all_classes_in_test():
    dataset = classification_dataset()
    fold = build_expanding_folds(120)[0]

    altered_target = list(dataset.target)
    for index in range(fold.test_start, fold.test_end):
        altered_target[index] = "UP"

    altered = MLDataset(
        dataset_id="missing-test-class",
        source="synthetic test data",
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        features=dataset.features,
        target=tuple(altered_target),
        task="classification",
    )

    with pytest.raises(ValueError, match="test window"):
        evaluate_execution_fold(
            altered,
            fold,
            algorithm="logistic_regression",
            seed=28,
            dates=tuple(f"row-{index}" for index in range(120)),
        )
