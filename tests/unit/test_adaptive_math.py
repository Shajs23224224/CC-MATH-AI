"""Behavioral checks for adaptive strategy selection and safe expression synthesis."""

import numpy as np
import pytest

from core.contracts.adaptive_math import (
    AdaptiveCandidateEvidence,
    AdaptiveMathConfig,
    AdaptiveMathReport,
    ExpressionNode,
)
from core.contracts.ml import MLDataset
from ml import AdaptiveMathEngine, JSONLExperimentStore, predict_adaptive_candidate
from ml.errors import MLInputError


def make_dataset(*, relation: str = "linear", rows: int = 100) -> MLDataset:
    x = np.linspace(-4.0, 4.0, rows)
    if relation == "linear":
        y = 2.0 * x + 3.0
    elif relation == "quadratic":
        y = x**2 - 0.5 * x + 2.0
    else:
        y = np.sin(x)
    return MLDataset(
        dataset_id=f"synthetic-adaptive-{relation}-v1",
        source=f"synthetic fixture with known {relation} relation",
        feature_names=("x",),
        target_name="y",
        features=tuple((float(value),) for value in x),
        target=tuple(float(value) for value in y),
        task="regression",
    )


def small_config(**updates: object) -> AdaptiveMathConfig:
    values: dict[str, object] = {
        "random_seed": 28,
        "population_size": 8,
        "generations": 3,
        "max_evaluations": 40,
        "max_degree": 3,
        "max_terms": 32,
    }
    values.update(updates)
    return AdaptiveMathConfig(**values)


def test_adaptive_engine_finds_held_out_linear_relation() -> None:
    report = AdaptiveMathEngine(small_config()).analyze(make_dataset())
    assert report.selected_candidate.test_rmse is not None
    assert report.selected_candidate.test_rmse < 1e-8
    assert report.test_used_for_selection is False
    assert report.selected_candidate.test_used_for_selection is False
    assert report.test_start_index == report.validation_end_index
    assert report.test_end_index == 100
    predictions = predict_adaptive_candidate(report.selected_candidate, ((1.0,), (2.0,)))
    assert predictions == pytest.approx((5.0, 7.0), abs=1e-7)
    assert report.formal_proof is False
    assert any("RMSE independently recomputed" in item for item in report.verification_checks)


def test_adaptive_strategy_selection_detects_quadratic_structure() -> None:
    report = AdaptiveMathEngine(small_config()).analyze(make_dataset(relation="quadratic"))
    assert report.selected_candidate.test_rmse is not None
    assert report.selected_candidate.test_rmse < 1e-5
    assert report.selected_candidate.validation_rmse < 1e-5
    assert report.selected_candidate.strategy in {"polynomial", "evolutionary"}


def test_evolutionary_candidates_are_real_bounded_and_reproducible() -> None:
    config = small_config(random_seed=72)
    dataset = make_dataset(relation="sine")
    first = AdaptiveMathEngine(config).analyze(dataset)
    second = AdaptiveMathEngine(config).analyze(dataset)
    assert "evolutionary" in first.strategy_scores
    assert first.candidates_evaluated <= config.max_evaluations
    assert first.generations_completed > 0
    assert first.selected_candidate.candidate_id == second.selected_candidate.candidate_id
    assert first.selected_candidate.test_rmse == second.selected_candidate.test_rmse
    assert all(
        candidate.test_used_for_selection is False for candidate in first.candidate_leaderboard
    )


def test_report_and_structured_candidate_survive_json_round_trip() -> None:
    report = AdaptiveMathEngine(small_config()).analyze(make_dataset())
    recovered = AdaptiveMathReport.model_validate_json(report.model_dump_json())
    assert recovered == report
    assert predict_adaptive_candidate(recovered.selected_candidate, ((0.0,),))[0] == pytest.approx(
        3.0, abs=1e-6
    )


def test_experiment_store_persists_adaptive_run(tmp_path) -> None:
    store = JSONLExperimentStore(tmp_path / "adaptive" / "runs.jsonl")
    report = AdaptiveMathEngine(small_config()).analyze(make_dataset(), store=store)
    records = store.read_all()
    assert len(records) == 1
    assert records[0].kind == "adaptive_math_discovery"
    assert records[0].subject_id == report.selected_candidate.candidate_id
    assert records[0].results["test_used_for_selection"] is False


def test_rejects_dataset_outside_computation_budget() -> None:
    engine = AdaptiveMathEngine(small_config(max_observations=30))
    with pytest.raises(MLInputError, match="max_observations"):
        engine.analyze(make_dataset(rows=40))


def test_invalid_generated_division_is_rejected_at_prediction_time() -> None:
    tree = ExpressionNode(
        op="div",
        children=(
            ExpressionNode(op="feature", feature_index=0),
            ExpressionNode(op="constant", constant=0.0),
        ),
    )
    candidate = AdaptiveCandidateEvidence(
        candidate_id="division-domain-test",
        dataset_id="synthetic-domain-test",
        strategy="evolutionary",
        feature_names=("x",),
        representation="y ≈ x / 0",
        expression_tree=tree,
        coefficients=(1.0,),
        intercept=0.0,
        complexity=4,
        fit_samples=20,
        fit_rmse=1.0,
        validation_rmse=1.0,
        selection_score=1.0004,
        limitations=("synthetic domain test",),
    )
    with pytest.raises(MLInputError, match="division denominator"):
        predict_adaptive_candidate(candidate, ((1.0,), (2.0,)))


def test_rejects_classification_in_math_regression_coordinator() -> None:
    dataset = MLDataset(
        dataset_id="synthetic-classes",
        source="synthetic test fixture",
        feature_names=("x",),
        target_name="label",
        features=tuple((float(index),) for index in range(24)),
        target=tuple("even" if index % 2 == 0 else "odd" for index in range(24)),
        task="classification",
    )
    with pytest.raises(MLInputError, match="currently handles regression"):
        AdaptiveMathEngine(small_config()).analyze(dataset)
