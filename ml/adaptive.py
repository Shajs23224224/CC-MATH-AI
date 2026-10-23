"""Adaptive mathematical strategy selection and bounded evolutionary synthesis.

The coordinator compares a linear baseline, bounded polynomial models and a seeded
expression-tree search. Validation chooses the candidate; the chronological test tail
is evaluated only once after selection. Candidate trees are data interpreted by an
allow-listed evaluator, never executable source code.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from core.contracts.adaptive_math import (
    AdaptiveCandidateEvidence,
    AdaptiveMathConfig,
    AdaptiveMathReport,
)
from core.contracts.ml import ExperimentRecord, MLDataset, PatternRecord

from .errors import ExperimentPersistenceError, MLInputError, MLTrainingError
from .evolution import (
    InvalidExpression,
    Node,
    crossover_trees,
    evaluate_tree,
    from_contract,
    mutate_tree,
    node_count,
    node_depth,
    random_tree,
    render_tree,
    to_contract,
)
from .patterns import discover_patterns
from .store import JSONLExperimentStore

FloatArray = NDArray[np.float64]
Strategy = Literal["linear", "polynomial", "evolutionary"]
_LIMITATIONS = (
    "The report is experimental evidence, not a formal mathematical proof.",
    "Rows must already be ordered chronologically; the coordinator cannot repair bad source "
    "lineage.",
    "The test tail is held out from candidate and strategy selection.",
    "Evolution explores only the configured allow-listed expression grammar and budgets.",
    "Association records are exploratory and do not establish causality.",
)


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    candidate_id: str
    strategy: Strategy
    representation: str
    degree: int | None
    tree: Node | None
    coefficients: tuple[float, ...]
    intercept: float
    complexity: int
    fit_rmse: float
    validation_rmse: float
    selection_score: float


def _matrix(rows: Sequence[Sequence[float]], width: int | None = None) -> FloatArray:
    try:
        values = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MLInputError("features must be a rectangular numeric matrix") from exc
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise MLInputError("features must be a non-empty two-dimensional matrix")
    if width is not None and values.shape[1] != width:
        raise MLInputError(f"expected {width} features per row")
    if not np.all(np.isfinite(values)):
        raise MLInputError("features must be finite")
    return values


def _rmse(actual: FloatArray, predicted: FloatArray) -> float:
    result = float(np.sqrt(np.mean(np.square(predicted - actual))))
    if not math.isfinite(result):
        raise MLTrainingError("candidate produced non-finite RMSE")
    return result


def _regression_metrics(actual: FloatArray, predicted: FloatArray) -> tuple[float, float, float]:
    errors = predicted - actual
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    total = float(np.sum((actual - float(np.mean(actual))) ** 2))
    r2 = 0.0 if total <= 0.0 else float(1.0 - np.sum(errors**2) / total)
    if not all(math.isfinite(value) for value in (rmse, mae, r2)):
        raise MLTrainingError("regression evaluation produced non-finite metrics")
    return rmse, mae, r2


def _fingerprint(dataset_id: str, matrix: FloatArray, target: FloatArray, seed: int) -> str:
    # Fingerprint only the development prefix, never the final test tail.
    payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "features": matrix.tolist(),
            "target": target.tolist(),
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_id(fingerprint: str, strategy: Strategy, representation: str) -> str:
    return hashlib.sha256(f"{fingerprint}|{strategy}|{representation}".encode()).hexdigest()[:24]


def _format_equation(intercept: float, terms: Sequence[str], coefficients: Sequence[float]) -> str:
    parts = [f"{intercept:.8g}"]
    for term, coefficient in zip(terms, coefficients, strict=True):
        if abs(coefficient) >= 1e-10:
            parts.append(f"({coefficient:+.8g} * {term})")
    return "y ≈ " + " ".join(parts)


def _evaluate_dense_candidate(
    strategy: Literal["linear", "polynomial"],
    degree: int | None,
    matrix_train: FloatArray,
    target_train: FloatArray,
    matrix_validation: FloatArray,
    target_validation: FloatArray,
    feature_names: tuple[str, ...],
    fingerprint: str,
    config: AdaptiveMathConfig,
) -> _ScoredCandidate:
    if strategy == "linear":
        train_design = matrix_train
        validation_design = matrix_validation
        terms = feature_names
        degree_for_candidate = None
        representation_prefix = "linear"
    else:
        assert degree is not None
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        train_design = transformer.fit_transform(matrix_train)
        validation_design = transformer.transform(matrix_validation)
        if not np.all(np.isfinite(train_design)) or not np.all(np.isfinite(validation_design)):
            raise MLTrainingError("polynomial expansion overflowed")
        terms = tuple(str(value) for value in transformer.get_feature_names_out(feature_names))
        degree_for_candidate = degree
        representation_prefix = f"polynomial-degree-{degree}"
    estimator = LinearRegression()
    estimator.fit(train_design, target_train)
    fit_prediction = np.asarray(estimator.predict(train_design), dtype=np.float64)
    validation_prediction = np.asarray(estimator.predict(validation_design), dtype=np.float64)
    if not np.all(np.isfinite(fit_prediction)) or not np.all(np.isfinite(validation_prediction)):
        raise MLTrainingError("dense candidate produced non-finite predictions")
    coefficients = tuple(float(value) for value in np.asarray(estimator.coef_).reshape(-1))
    intercept = float(estimator.intercept_)
    complexity = max(1, 1 + sum(abs(value) > 1e-9 for value in coefficients))
    fit_rmse = _rmse(target_train, fit_prediction)
    validation_rmse = _rmse(target_validation, validation_prediction)
    score = validation_rmse + config.complexity_penalty * complexity
    representation = f"{representation_prefix}: " + _format_equation(intercept, terms, coefficients)
    return _ScoredCandidate(
        candidate_id=_candidate_id(fingerprint, strategy, representation_prefix),
        strategy=strategy,
        representation=representation,
        degree=degree_for_candidate,
        tree=None,
        coefficients=coefficients,
        intercept=intercept,
        complexity=complexity,
        fit_rmse=fit_rmse,
        validation_rmse=validation_rmse,
        selection_score=score,
    )


def _evaluate_evolutionary_candidate(
    tree: Node,
    matrix_train: FloatArray,
    target_train: FloatArray,
    matrix_validation: FloatArray,
    target_validation: FloatArray,
    feature_names: tuple[str, ...],
    fingerprint: str,
    config: AdaptiveMathConfig,
) -> _ScoredCandidate:
    if node_count(tree) > config.max_nodes or node_depth(tree) > config.max_depth:
        raise InvalidExpression("candidate exceeds configured depth/node limits")
    train_value = evaluate_tree(
        tree, matrix_train, max_depth=config.max_depth, max_nodes=config.max_nodes
    )
    validation_value = evaluate_tree(
        tree, matrix_validation, max_depth=config.max_depth, max_nodes=config.max_nodes
    )
    if float(np.ptp(train_value)) <= 1e-12:
        raise InvalidExpression("constant expression rejected as an uninformative genome")
    calibration = LinearRegression()
    calibration.fit(train_value.reshape(-1, 1), target_train)
    fit_prediction = np.asarray(calibration.predict(train_value.reshape(-1, 1)), dtype=np.float64)
    validation_prediction = np.asarray(
        calibration.predict(validation_value.reshape(-1, 1)), dtype=np.float64
    )
    if not np.all(np.isfinite(fit_prediction)) or not np.all(np.isfinite(validation_prediction)):
        raise InvalidExpression("calibrated expression produced non-finite output")
    scale = float(np.asarray(calibration.coef_).reshape(-1)[0])
    intercept = float(calibration.intercept_)
    complexity = node_count(tree) + 2
    expression = render_tree(tree, feature_names)
    representation = f"y ≈ ({scale:.8g} * {expression}) + {intercept:.8g}"
    return _ScoredCandidate(
        candidate_id=_candidate_id(fingerprint, "evolutionary", expression),
        strategy="evolutionary",
        representation=representation,
        degree=None,
        tree=tree,
        coefficients=(scale,),
        intercept=intercept,
        complexity=complexity,
        fit_rmse=_rmse(target_train, fit_prediction),
        validation_rmse=_rmse(target_validation, validation_prediction),
        selection_score=_rmse(target_validation, validation_prediction)
        + config.complexity_penalty * complexity,
    )


def _candidate_contract(
    scored: _ScoredCandidate,
    dataset_id: str,
    feature_names: tuple[str, ...],
    fit_samples: int | None = None,
    fit_rmse: float | None = None,
) -> AdaptiveCandidateEvidence:
    return AdaptiveCandidateEvidence(
        candidate_id=scored.candidate_id,
        dataset_id=dataset_id,
        strategy=scored.strategy,
        feature_names=feature_names,
        representation=scored.representation,
        expression_tree=to_contract(scored.tree) if scored.tree is not None else None,
        degree=scored.degree,
        coefficients=scored.coefficients,
        intercept=scored.intercept,
        complexity=scored.complexity,
        fit_samples=fit_samples if fit_samples is not None else 1,
        fit_rmse=fit_rmse if fit_rmse is not None else scored.fit_rmse,
        validation_rmse=scored.validation_rmse,
        selection_score=scored.selection_score,
        limitations=_LIMITATIONS,
    )


def _fit_final_candidate(
    candidate: _ScoredCandidate,
    matrix_development: FloatArray,
    target_development: FloatArray,
    matrix_test: FloatArray,
    target_test: FloatArray,
    feature_names: tuple[str, ...],
    dataset_id: str,
    config: AdaptiveMathConfig,
) -> AdaptiveCandidateEvidence:
    if candidate.strategy == "evolutionary":
        assert candidate.tree is not None
        development_value = evaluate_tree(
            candidate.tree,
            matrix_development,
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
        )
        test_value = evaluate_tree(
            candidate.tree,
            matrix_test,
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
        )
        estimator = LinearRegression()
        estimator.fit(development_value.reshape(-1, 1), target_development)
        development_prediction = np.asarray(
            estimator.predict(development_value.reshape(-1, 1)), dtype=np.float64
        )
        test_prediction = np.asarray(estimator.predict(test_value.reshape(-1, 1)), dtype=np.float64)
        scale = float(np.asarray(estimator.coef_).reshape(-1)[0])
        intercept = float(estimator.intercept_)
        expression = render_tree(candidate.tree, feature_names)
        representation = f"y ≈ ({scale:.8g} * {expression}) + {intercept:.8g}"
        final_coefficients = (scale,)
        final_intercept = intercept
    else:
        degree = candidate.degree if candidate.strategy == "polynomial" else 1
        if candidate.strategy == "linear":
            development_design = matrix_development
            test_design = matrix_test
            final_terms = feature_names
        else:
            transformer = PolynomialFeatures(degree=degree, include_bias=False)
            development_design = transformer.fit_transform(matrix_development)
            test_design = transformer.transform(matrix_test)
            if not np.all(np.isfinite(development_design)) or not np.all(np.isfinite(test_design)):
                raise MLTrainingError("final polynomial expansion overflowed")
            final_terms = tuple(
                str(value) for value in transformer.get_feature_names_out(feature_names)
            )
        estimator = LinearRegression().fit(development_design, target_development)
        development_prediction = np.asarray(estimator.predict(development_design), dtype=np.float64)
        test_prediction = np.asarray(estimator.predict(test_design), dtype=np.float64)
        final_coefficients = tuple(
            float(value) for value in np.asarray(estimator.coef_).reshape(-1)
        )
        final_intercept = float(estimator.intercept_)
        representation = (
            f"{candidate.strategy}"
            + (f"-degree-{degree}" if candidate.strategy == "polynomial" else "")
            + ": "
            + _format_equation(final_intercept, final_terms, final_coefficients)
        )
    if (
        not np.all(np.isfinite(development_prediction))
        or not np.all(np.isfinite(test_prediction))
        or test_prediction.shape[0] != target_test.shape[0]
    ):
        raise MLTrainingError("final candidate failed finite-output or shape verification")

    test_rmse, test_mae, test_r2 = _regression_metrics(target_test, test_prediction)
    # Independent scalar recomputation acts as a differential numerical sanity check.
    reference_rmse = math.sqrt(
        math.fsum(
            (float(predicted) - float(actual)) ** 2
            for actual, predicted in zip(target_test, test_prediction, strict=True)
        )
        / len(target_test)
    )
    if not math.isclose(test_rmse, reference_rmse, rel_tol=1e-10, abs_tol=1e-10):
        raise MLTrainingError("independent RMSE recomputation disagreed with the metric engine")
    final_fit_rmse = _rmse(target_development, development_prediction)
    return AdaptiveCandidateEvidence(
        candidate_id=candidate.candidate_id,
        dataset_id=dataset_id,
        strategy=candidate.strategy,
        feature_names=feature_names,
        representation=representation,
        expression_tree=to_contract(candidate.tree) if candidate.tree is not None else None,
        degree=candidate.degree,
        coefficients=final_coefficients,
        intercept=final_intercept,
        complexity=candidate.complexity,
        fit_samples=len(target_development),
        fit_rmse=final_fit_rmse,
        validation_rmse=candidate.validation_rmse,
        selection_score=candidate.selection_score,
        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2,
        test_samples=len(target_test),
        test_used_for_selection=False,
        status="EXPERIMENTAL",
        limitations=_LIMITATIONS,
    )


def predict_adaptive_candidate(
    candidate: AdaptiveCandidateEvidence,
    rows: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Safely predict from a stored adaptive candidate without dynamic code execution."""
    matrix = _matrix(rows, len(candidate.feature_names))
    if candidate.strategy == "evolutionary":
        if candidate.expression_tree is None:
            raise MLInputError("evolutionary candidate has no structured expression tree")
        tree = from_contract(candidate.expression_tree)
        try:
            values = evaluate_tree(tree, matrix, max_depth=10, max_nodes=127)
        except InvalidExpression as exc:
            raise MLInputError(str(exc)) from exc
        predictions = candidate.coefficients[0] * values + candidate.intercept
    elif candidate.strategy == "linear":
        if len(candidate.coefficients) != matrix.shape[1]:
            raise MLInputError("linear candidate coefficient count does not match features")
        predictions = matrix @ np.asarray(candidate.coefficients) + candidate.intercept
    else:
        assert candidate.degree is not None
        term_count = math.comb(len(candidate.feature_names) + candidate.degree, candidate.degree) - 1
        if term_count > 512:
            raise MLInputError(
                "stored polynomial candidate exceeds the hard 512-term execution limit"
            )
        transformed = PolynomialFeatures(degree=candidate.degree, include_bias=False).fit_transform(
            matrix
        )
        if transformed.shape[1] != len(candidate.coefficients):
            raise MLInputError("polynomial candidate representation is incompatible with input")
        predictions = transformed @ np.asarray(candidate.coefficients) + candidate.intercept
    if not np.all(np.isfinite(predictions)):
        raise MLInputError("candidate returned non-finite predictions")
    return tuple(float(value) for value in predictions)


class AdaptiveMathEngine:
    """Select and verify candidate mathematical strategies from chronological data.

    Current scope is supervised real-valued regression. Classification remains available
    through ClassicalMLModel; this coordinator deliberately rejects unsupported tasks.
    """

    def __init__(self, config: AdaptiveMathConfig | None = None) -> None:
        self.config = config or AdaptiveMathConfig()

    def analyze(
        self,
        dataset: MLDataset,
        *,
        store: JSONLExperimentStore | None = None,
    ) -> AdaptiveMathReport:
        if dataset.task != "regression":
            raise MLInputError(
                "AdaptiveMathEngine currently handles regression; "
                "use ClassicalMLModel for classification"
            )
        matrix = _matrix(dataset.features, len(dataset.feature_names))
        target = np.asarray(dataset.target, dtype=np.float64)
        total, width = matrix.shape
        if total > self.config.max_observations:
            raise MLInputError(
                f"dataset has {total} rows; max_observations is {self.config.max_observations}"
            )
        test_size = max(3, int(math.ceil(total * self.config.test_fraction)))
        remaining = total - test_size
        validation_size = max(3, int(math.ceil(remaining * self.config.validation_fraction)))
        train_end = total - test_size - validation_size
        validation_end = train_end + validation_size
        if train_end < 12:
            raise MLInputError(
                "adaptive analysis requires at least 12 training rows plus validation and test rows"
            )
        x_train, y_train = matrix[:train_end], target[:train_end]
        x_validation, y_validation = (
            matrix[train_end:validation_end],
            target[train_end:validation_end],
        )
        x_development, y_development = matrix[:validation_end], target[:validation_end]
        x_test, y_test = matrix[validation_end:], target[validation_end:]
        fingerprint = _fingerprint(
            dataset.dataset_id, x_development, y_development, self.config.random_seed
        )
        rng = np.random.default_rng(self.config.random_seed)

        candidates: list[_ScoredCandidate] = []
        candidate_count = 0
        rejected_count = 0
        try:
            candidates.append(
                _evaluate_dense_candidate(
                    "linear",
                    None,
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    dataset.feature_names,
                    fingerprint,
                    self.config,
                )
            )
            candidate_count += 1
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            candidate_count += 1
            rejected_count += 1
            raise MLTrainingError("linear baseline could not be fitted") from exc

        for degree in range(1, self.config.max_degree + 1):
            candidate_count += 1
            term_count = math.comb(width + degree, degree) - 1
            if term_count > self.config.max_terms:
                rejected_count += 1
                continue
            try:
                candidates.append(
                    _evaluate_dense_candidate(
                        "polynomial",
                        degree,
                        x_train,
                        y_train,
                        x_validation,
                        y_validation,
                        dataset.feature_names,
                        fingerprint,
                        self.config,
                    )
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, MLTrainingError):
                rejected_count += 1

        available_evaluations = self.config.max_evaluations - candidate_count
        generations_to_run = min(
            self.config.generations, max(0, available_evaluations // self.config.population_size)
        )
        population = [
            random_tree(rng, width, self.config.max_depth, self.config.max_nodes)
            for _ in range(self.config.population_size)
        ]
        generations_completed = 0
        evolutionary_candidates: list[_ScoredCandidate] = []
        for _generation in range(generations_to_run):
            scored_population: list[tuple[Node, _ScoredCandidate]] = []
            for tree in population:
                if candidate_count >= self.config.max_evaluations:
                    break
                candidate_count += 1
                try:
                    scored = _evaluate_evolutionary_candidate(
                        tree,
                        x_train,
                        y_train,
                        x_validation,
                        y_validation,
                        dataset.feature_names,
                        fingerprint,
                        self.config,
                    )
                    evolutionary_candidates.append(scored)
                    scored_population.append((tree, scored))
                except (InvalidExpression, ValueError, FloatingPointError, np.linalg.LinAlgError):
                    rejected_count += 1
            generations_completed += 1
            if len(scored_population) < 2:
                population = [
                    random_tree(rng, width, self.config.max_depth, self.config.max_nodes)
                    for _ in range(self.config.population_size)
                ]
                continue
            scored_population.sort(
                key=lambda pair: (pair[1].selection_score, pair[1].complexity, pair[1].candidate_id)
            )
            parent_count = max(2, min(len(scored_population), self.config.population_size // 2))
            parents = [item[0] for item in scored_population[:parent_count]]
            elite_count = min(len(parents), max(2, self.config.population_size // 4))
            next_population = parents[:elite_count]
            while len(next_population) < self.config.population_size:
                choice = float(rng.random())
                first_parent = parents[int(rng.integers(0, len(parents)))]
                if choice < self.config.crossover_rate and len(parents) > 1:
                    second_parent = parents[int(rng.integers(0, len(parents)))]
                    child = crossover_trees(
                        first_parent,
                        second_parent,
                        rng,
                        width,
                        self.config.max_depth,
                        self.config.max_nodes,
                    )
                elif choice < self.config.crossover_rate + self.config.mutation_rate:
                    child = mutate_tree(
                        first_parent, rng, width, self.config.max_depth, self.config.max_nodes
                    )
                else:
                    child = random_tree(rng, width, self.config.max_depth, self.config.max_nodes)
                if (
                    node_count(child) > self.config.max_nodes
                    or node_depth(child) > self.config.max_depth
                ):
                    child = random_tree(rng, width, self.config.max_depth, self.config.max_nodes)
                next_population.append(child)
            population = next_population

        candidates.extend(evolutionary_candidates)
        if not candidates:
            raise MLTrainingError("adaptive search could not produce any valid candidate")
        candidates.sort(
            key=lambda item: (
                item.selection_score,
                item.complexity,
                item.strategy,
                item.candidate_id,
            )
        )
        winner: _ScoredCandidate | None = None
        selected: AdaptiveCandidateEvidence | None = None
        final_safety_rejections = 0
        checked_candidate_ids: set[str] = set()
        for candidate in candidates:
            if candidate.candidate_id in checked_candidate_ids:
                continue
            checked_candidate_ids.add(candidate.candidate_id)
            try:
                final_candidate = _fit_final_candidate(
                    candidate,
                    x_development,
                    y_development,
                    x_test,
                    y_test,
                    dataset.feature_names,
                    dataset.dataset_id,
                    self.config,
                )
            except (InvalidExpression, MLTrainingError, ValueError, FloatingPointError):
                # Safety/domain validity is a hard constraint, not a test-performance score.
                final_safety_rejections += 1
                continue
            winner = candidate
            selected = final_candidate
            break
        if winner is None or selected is None:
            raise MLTrainingError("every candidate failed final domain or numerical safety checks")

        # Pattern discovery only sees the training prefix; it cannot inspect validation/test tails.
        training_dataset_id = f"{dataset.dataset_id[:180]}-train-prefix"
        training_dataset = MLDataset(
            dataset_id=training_dataset_id,
            source=f"{dataset.source[:450]} (training prefix only)",
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            features=tuple(tuple(float(v) for v in row) for row in x_train),
            target=tuple(float(v) for v in y_train),
            task="regression",
        )
        patterns: tuple[PatternRecord, ...] = discover_patterns(
            training_dataset, min_abs_correlation=0.4
        )

        best_by_id: dict[str, _ScoredCandidate] = {}
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.selection_score, item.complexity, item.strategy, item.candidate_id
            ),
        ):
            best_by_id.setdefault(candidate.candidate_id, candidate)
        top_scored = list(best_by_id.values())[:8]
        top_contracts = [
            selected if candidate.candidate_id == selected.candidate_id
            else _candidate_contract(
                candidate, dataset.dataset_id, dataset.feature_names, fit_samples=train_end
            )
            for candidate in top_scored
        ]
        if selected.candidate_id not in {candidate.candidate_id for candidate in top_contracts}:
            top_contracts[-1] = selected
        leaderboard = tuple(top_contracts)
        strategy_scores: dict[str, float] = {}
        for candidate in candidates:
            existing = strategy_scores.get(candidate.strategy)
            if existing is None or candidate.selection_score < existing:
                strategy_scores[candidate.strategy] = candidate.selection_score

        verification_checks = (
            "chronological train/validation/test boundaries checked",
            "candidate re-evaluated from its structured allow-listed representation",
            "final test predictions have the expected shape and are finite",
            "RMSE independently recomputed with scalar math.fsum",
            "test tail excluded from strategy and candidate selection",
            "generated candidates are never executed as Python source",
            "discovered patterns use the training prefix only",
        )
        report = AdaptiveMathReport(
            dataset_id=dataset.dataset_id,
            selected_candidate=selected,
            candidate_leaderboard=leaderboard,
            patterns=patterns,
            train_end_index=train_end,
            validation_end_index=validation_end,
            test_start_index=validation_end,
            test_end_index=total,
            training_samples=train_end,
            validation_samples=validation_size,
            test_samples=test_size,
            candidates_evaluated=candidate_count,
            candidates_rejected=rejected_count,
            final_safety_rejections=final_safety_rejections,
            generations_completed=generations_completed,
            random_seed=self.config.random_seed,
            strategy_scores=strategy_scores,
            selected_reason=(
                f"Selected {winner.strategy} as the lowest-validation-score candidate that passed "
                "final domain/numerical safety checks. Validation score is RMSE plus the "
                "configured "
                "complexity penalty; test metrics do not rank candidates."
            ),
            verification_checks=verification_checks,
            limitations=_LIMITATIONS,
        )
        if store is not None:
            record = ExperimentRecord(
                kind="adaptive_math_discovery",
                subject_id=selected.candidate_id,
                status="EXPERIMENTAL",
                configuration={
                    "seed": self.config.random_seed,
                    "max_degree": self.config.max_degree,
                    "max_terms": self.config.max_terms,
                    "population_size": self.config.population_size,
                    "generations_requested": self.config.generations,
                    "max_evaluations": self.config.max_evaluations,
                    "complexity_penalty": self.config.complexity_penalty,
                    "split": "chronological_train_validation_test",
                    "strategy": selected.strategy,
                },
                results={
                    "fit_rmse": selected.fit_rmse,
                    "validation_rmse": selected.validation_rmse,
                    "test_rmse": selected.test_rmse if selected.test_rmse is not None else 0.0,
                    "test_mae": selected.test_mae if selected.test_mae is not None else 0.0,
                    "candidates_evaluated": candidate_count,
                    "candidates_rejected": rejected_count,
                    "final_safety_rejections": final_safety_rejections,
                    "generations_completed": generations_completed,
                    "test_used_for_selection": False,
                },
                limitations=_LIMITATIONS,
            )
            try:
                store.append(record)
            except ExperimentPersistenceError:
                raise
            except OSError as exc:
                raise ExperimentPersistenceError("adaptive result could not be persisted") from exc
        return report


__all__ = ["AdaptiveMathEngine", "predict_adaptive_candidate"]
