"""Typed contracts for bounded adaptive mathematical discovery.

Candidates are data, not executable code. Expression trees are interpreted by an
allow-listed numerical evaluator and are subject to runtime depth/node budgets.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ml import CandidateStatus, PatternRecord

ExpressionOperator = Literal[
    "feature", "constant", "add", "sub", "mul", "div", "sin", "cos", "tanh", "abs"
]
AdaptiveStrategy = Literal["linear", "polynomial", "evolutionary"]


class ExpressionNode(BaseModel):
    """Safe mathematical expression tree node; never stores source code."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    op: ExpressionOperator
    feature_index: int | None = Field(default=None, ge=0, le=63)
    constant: float | None = None
    children: tuple[ExpressionNode, ...] = Field(default_factory=tuple, max_length=2)

    @model_validator(mode="after")
    def validate_node_shape(self) -> ExpressionNode:
        unary = {"sin", "cos", "tanh", "abs"}
        binary = {"add", "sub", "mul", "div"}
        if self.op == "feature":
            if self.feature_index is None or self.constant is not None or self.children:
                raise ValueError("feature nodes require only a feature_index")
        elif self.op == "constant":
            if self.constant is None or self.feature_index is not None or self.children:
                raise ValueError("constant nodes require only a finite constant")
            if not math.isfinite(self.constant):
                raise ValueError("constant values must be finite")
        elif self.op in unary:
            if len(self.children) != 1 or self.feature_index is not None or self.constant is not None:
                raise ValueError("unary operators require exactly one child")
        elif self.op in binary:
            if len(self.children) != 2 or self.feature_index is not None or self.constant is not None:
                raise ValueError("binary operators require exactly two children")
        return self


ExpressionNode.model_rebuild()


class AdaptiveMathConfig(BaseModel):
    """Reproducible, CPU-first controls for strategy selection and evolutionary search."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    random_seed: int = Field(default=28, ge=0, le=2**32 - 1)
    validation_fraction: float = Field(default=0.2, gt=0.05, lt=0.4)
    test_fraction: float = Field(default=0.2, gt=0.05, lt=0.4)
    max_degree: int = Field(default=3, ge=1, le=4)
    max_terms: int = Field(default=128, ge=1, le=512)
    population_size: int = Field(default=16, ge=4, le=64)
    generations: int = Field(default=6, ge=1, le=16)
    max_evaluations: int = Field(default=96, ge=8, le=1024)
    max_nodes: int = Field(default=31, ge=3, le=127)
    max_depth: int = Field(default=6, ge=1, le=10)
    mutation_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    complexity_penalty: float = Field(default=1e-4, ge=0.0, le=1.0)
    max_observations: int = Field(default=10000, ge=20, le=100000)

    @model_validator(mode="after")
    def validate_budgets(self) -> AdaptiveMathConfig:
        if self.validation_fraction + self.test_fraction >= 0.6:
            raise ValueError("validation and test fractions must sum to less than 0.6")
        if self.max_evaluations < self.population_size + self.max_degree + 1:
            raise ValueError(
                "max_evaluations must allow the initial population and baseline candidates"
            )
        if self.max_evaluations * self.max_nodes * self.max_observations > 50_000_000:
            raise ValueError(
                "combined candidate, node and observation budget exceeds 50 million work units"
            )
        return self


class AdaptiveCandidateEvidence(BaseModel):
    """A selected or leaderboard candidate with reproducible representation and evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    candidate_id: str = Field(min_length=1, max_length=64)
    dataset_id: str = Field(min_length=1, max_length=200)
    strategy: AdaptiveStrategy
    feature_names: tuple[str, ...] = Field(min_length=1, max_length=64)
    representation: str = Field(min_length=1, max_length=2000)
    expression_tree: ExpressionNode | None = None
    degree: int | None = Field(default=None, ge=1, le=4)
    coefficients: tuple[float, ...] = Field(min_length=1, max_length=512)
    intercept: float
    complexity: int = Field(ge=1, le=512)
    fit_samples: int = Field(ge=1)
    fit_rmse: float = Field(ge=0.0)
    validation_rmse: float = Field(ge=0.0)
    selection_score: float = Field(ge=0.0)
    test_rmse: float | None = Field(default=None, ge=0.0)
    test_mae: float | None = Field(default=None, ge=0.0)
    test_r2: float | None = None
    test_samples: int = Field(default=0, ge=0)
    test_used_for_selection: Literal[False] = False
    status: CandidateStatus = "EXPERIMENTAL"
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_representation(self) -> AdaptiveCandidateEvidence:
        if self.strategy == "evolutionary" and self.expression_tree is None:
            raise ValueError("evolutionary candidates require a structured expression tree")
        if self.strategy != "evolutionary" and self.expression_tree is not None:
            raise ValueError("only evolutionary candidates may carry an expression tree")
        if self.strategy == "polynomial" and self.degree is None:
            raise ValueError("polynomial candidates require a degree")
        if self.strategy != "polynomial" and self.degree is not None:
            raise ValueError("degree is only valid for polynomial candidates")
        if any(not math.isfinite(value) for value in self.coefficients):
            raise ValueError("candidate coefficients must be finite")
        if not all(
            math.isfinite(value)
            for value in (self.intercept, self.fit_rmse, self.validation_rmse, self.selection_score)
        ):
            raise ValueError("candidate metrics and intercept must be finite")
        if self.test_r2 is not None and not math.isfinite(self.test_r2):
            raise ValueError("test_r2 must be finite")
        if self.test_samples == 0 and any(
            value is not None for value in (self.test_rmse, self.test_mae, self.test_r2)
        ):
            raise ValueError("test metrics require a positive test sample count")
        if self.test_samples > 0 and (self.test_rmse is None or self.test_mae is None):
            raise ValueError("final candidate test metrics must include RMSE and MAE")
        return self


class AdaptiveMathReport(BaseModel):
    """Auditable adaptive strategy decision; experiments remain non-proofs."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    dataset_id: str = Field(min_length=1)
    selected_candidate: AdaptiveCandidateEvidence
    candidate_leaderboard: tuple[AdaptiveCandidateEvidence, ...] = Field(min_length=1, max_length=8)
    patterns: tuple[PatternRecord, ...] = ()
    train_end_index: int = Field(ge=1)
    validation_end_index: int = Field(ge=1)
    test_start_index: int = Field(ge=1)
    test_end_index: int = Field(ge=1)
    training_samples: int = Field(ge=1)
    validation_samples: int = Field(ge=1)
    test_samples: int = Field(ge=1)
    candidates_evaluated: int = Field(ge=1)
    candidates_rejected: int = Field(ge=0)
    generations_completed: int = Field(ge=0)
    random_seed: int = Field(ge=0)
    strategy_scores: dict[str, float]
    selected_reason: str = Field(min_length=1)
    verification_checks: tuple[str, ...] = Field(min_length=1)
    formal_proof: Literal[False] = False
    test_used_for_selection: Literal[False] = False
    status: CandidateStatus = "EXPERIMENTAL"
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_temporal_evidence(self) -> AdaptiveMathReport:
        if self.train_end_index != self.training_samples:
            raise ValueError("train_end_index must match training_samples")
        if self.validation_end_index - self.train_end_index != self.validation_samples:
            raise ValueError("validation split boundaries are inconsistent")
        if self.test_start_index != self.validation_end_index:
            raise ValueError("test must begin after the validation prefix")
        if self.test_end_index - self.test_start_index != self.test_samples:
            raise ValueError("test split boundary is inconsistent")
        if self.selected_candidate.test_samples != self.test_samples:
            raise ValueError("selected candidate must report the final held-out test size")
        if any(not math.isfinite(value) for value in self.strategy_scores.values()):
            raise ValueError("strategy scores must be finite")
        if self.selected_candidate.candidate_id not in {
            candidate.candidate_id for candidate in self.candidate_leaderboard
        }:
            raise ValueError("selected candidate must be present in the leaderboard")
        return self
