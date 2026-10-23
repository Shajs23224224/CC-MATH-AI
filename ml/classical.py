"""Reproducible, allow-listed classical ML with strict chronological evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.contracts.ml import (
    MLDataset,
    MLEvaluation,
    MLModelMetadata,
    MLPrediction,
    MLTrainingConfig,
)

from .errors import MLInputError, MLPredictionError, MLTrainingError

EstimatorFactory = Callable[[int], BaseEstimator]
_REGRESSORS: dict[str, EstimatorFactory] = {
    "linear_regression": lambda seed: LinearRegression(),
    "ridge": lambda seed: Ridge(),
    "lasso": lambda seed: Lasso(max_iter=20_000),
    "elastic_net": lambda seed: ElasticNet(max_iter=20_000, random_state=seed),
    "svr": lambda seed: SVR(),
    "knn_regressor": lambda seed: KNeighborsRegressor(),
    "decision_tree_regressor": lambda seed: DecisionTreeRegressor(random_state=seed),
    "random_forest_regressor": lambda seed: RandomForestRegressor(
        n_estimators=100, random_state=seed, n_jobs=1
    ),
    "gradient_boosting_regressor": lambda seed: GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed
    ),
}
_CLASSIFIERS: dict[str, EstimatorFactory] = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=2_000, random_state=seed),
    "svc": lambda seed: SVC(probability=True, random_state=seed),
    "knn_classifier": lambda seed: KNeighborsClassifier(),
    "decision_tree_classifier": lambda seed: DecisionTreeClassifier(random_state=seed),
    "random_forest_classifier": lambda seed: RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=1
    ),
    "gradient_boosting_classifier": lambda seed: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed
    ),
}
_BOOSTING_ALGORITHMS = {
    "gradient_boosting_classifier",
    "gradient_boosting_regressor",
}
_MAX_BOOSTING_ESTIMATORS = 1_000
_BOOSTING_PATIENCE = 10
_BOOSTING_MIN_ESTIMATORS = 10
_BOOSTING_VALIDATION_FRACTION = 0.15

_SCALED_MODELS = {
    "ridge",
    "lasso",
    "elastic_net",
    "svr",
    "knn_regressor",
    "logistic_regression",
    "svc",
    "knn_classifier",
}


def supported_algorithms() -> dict[str, tuple[str, ...]]:
    """Return the explicit registry of supported task/model combinations."""
    return {
        "regression": tuple(sorted(_REGRESSORS)),
        "classification": tuple(sorted(_CLASSIFIERS)),
    }


def _as_matrix(rows: Sequence[Sequence[float]], *, width: int | None = None) -> np.ndarray:
    try:
        matrix = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MLInputError("features must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise MLInputError("features must be a non-empty two-dimensional matrix")
    if width is not None and matrix.shape[1] != width:
        raise MLInputError(f"expected {width} features per row")
    if not np.all(np.isfinite(matrix)):
        raise MLInputError("features must contain only finite values")
    return matrix


def _fingerprint(features: np.ndarray, target: Sequence[float | str]) -> str:
    payload = json.dumps(
        {"features": features.tolist(), "target": list(target)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_gradient_boosting_iterations(
    estimator: GradientBoostingRegressor | GradientBoostingClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
) -> tuple[int, int]:
    """Select a tree count on a chronological validation suffix of the training prefix."""
    max_estimators = estimator.get_params(deep=False).get("n_estimators")
    if (
        isinstance(max_estimators, bool)
        or not isinstance(max_estimators, int)
        or not 1 <= max_estimators <= _MAX_BOOSTING_ESTIMATORS
    ):
        raise MLInputError(
            f"gradient boosting n_estimators must be between 1 and {_MAX_BOOSTING_ESTIMATORS}"
        )

    validation_size = max(2, int(math.ceil(len(x_train) * _BOOSTING_VALIDATION_FRACTION)))
    fit_end = len(x_train) - validation_size
    if fit_end < 8:
        raise MLInputError(
            "gradient boosting temporal early stopping requires at least eight inner-training "
            "samples and two validation samples"
        )
    x_fit, x_valid = x_train[:fit_end], x_train[fit_end:]
    y_fit, y_valid = y_train[:fit_end], y_train[fit_end:]
    if task == "classification" and len(np.unique(y_fit)) < 2:
        raise MLInputError(
            "gradient boosting inner training prefix must contain at least two target classes"
        )

    candidate = clone(estimator)
    candidate.set_params(warm_start=True, n_estimators=1)
    best_iteration = 1
    best_score = math.inf
    stale_iterations = 0
    minimum_iterations = min(_BOOSTING_MIN_ESTIMATORS, max_estimators)

    if isinstance(candidate, GradientBoostingRegressor):
        for iteration in range(1, max_estimators + 1):
            candidate.set_params(n_estimators=iteration)
            candidate.fit(x_fit, y_fit)
            predictions = np.asarray(candidate.predict(x_valid), dtype=np.float64)
            if not np.all(np.isfinite(predictions)):
                raise MLTrainingError(
                    "gradient boosting produced non-finite validation predictions"
                )
            score = float(mean_squared_error(y_valid.astype(np.float64), predictions))
            if score < best_score - 1e-12:
                best_score = score
                best_iteration = iteration
                stale_iterations = 0
            elif iteration >= minimum_iterations:
                stale_iterations += 1
            if iteration >= minimum_iterations and stale_iterations >= _BOOSTING_PATIENCE:
                break
    elif isinstance(candidate, GradientBoostingClassifier):
        for iteration in range(1, max_estimators + 1):
            candidate.set_params(n_estimators=iteration)
            candidate.fit(x_fit, y_fit)
            probabilities = np.asarray(candidate.predict_proba(x_valid), dtype=np.float64)
            if (
                probabilities.ndim != 2
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0.0)
                or np.any(probabilities > 1.0)
            ):
                raise MLTrainingError("gradient boosting produced invalid validation probabilities")
            score = float(log_loss(y_valid, probabilities, labels=candidate.classes_))
            if score < best_score - 1e-12:
                best_score = score
                best_iteration = iteration
                stale_iterations = 0
            elif iteration >= minimum_iterations:
                stale_iterations += 1
            if iteration >= minimum_iterations and stale_iterations >= _BOOSTING_PATIENCE:
                break
    else:
        raise MLInputError("unsupported estimator for gradient boosting early stopping")

    if not math.isfinite(best_score):
        raise MLTrainingError("gradient boosting could not select a finite validation score")
    return best_iteration, validation_size


class ClassicalMLModel:
    """Fit one allow-listed estimator and preserve auditable lineage and OOS metrics."""

    def __init__(self, config: MLTrainingConfig) -> None:
        self.config = config
        self._pipeline: Pipeline | None = None
        self.metadata: MLModelMetadata | None = None
        self.evaluation: MLEvaluation | None = None

    def fit(self, dataset: MLDataset) -> MLEvaluation:
        """Fit only the chronological training prefix and evaluate once on the held-out tail."""
        if dataset.task != self.config.task:
            raise MLInputError("dataset task must match model configuration")
        factories = _REGRESSORS if self.config.task == "regression" else _CLASSIFIERS
        if self.config.algorithm not in factories:
            raise MLInputError(
                "unsupported algorithm/task combination: "
                f"{self.config.algorithm}/{self.config.task}"
            )
        matrix = _as_matrix(dataset.features, width=len(dataset.feature_names))
        target = np.asarray(dataset.target)
        total = matrix.shape[0]
        test_size = max(2, int(math.ceil(total * self.config.test_fraction)))
        train_end = total - test_size
        if train_end < 8:
            raise MLInputError("chronological split must leave at least eight training samples")
        x_train, x_test = matrix[:train_end], matrix[train_end:]
        y_train, y_test = target[:train_end], target[train_end:]
        if self.config.task == "classification" and len(np.unique(y_train)) < 2:
            raise MLInputError("training prefix must contain at least two target classes")

        estimator = factories[self.config.algorithm](self.config.random_seed)
        supplied = dict(self.config.parameters)
        unsupported = set(supplied) - set(estimator.get_params(deep=False))
        if unsupported:
            raise MLInputError(f"unsupported model parameters: {', '.join(sorted(unsupported))}")

        boosting_details: dict[str, int | bool] = {}
        try:
            estimator.set_params(**supplied)
            if "random_state" in estimator.get_params(deep=False):
                estimator.set_params(random_state=self.config.random_seed)
            if "n_jobs" in estimator.get_params(deep=False):
                estimator.set_params(n_jobs=1)
            if self.config.algorithm in _BOOSTING_ALGORITHMS:
                if not isinstance(
                    estimator, (GradientBoostingClassifier, GradientBoostingRegressor)
                ):
                    raise MLInputError(
                        "gradient boosting registry returned an incompatible estimator"
                    )
                requested_max_estimators = estimator.get_params(deep=False).get("n_estimators")
                if (
                    isinstance(requested_max_estimators, bool)
                    or not isinstance(requested_max_estimators, int)
                    or not 1 <= requested_max_estimators <= _MAX_BOOSTING_ESTIMATORS
                ):
                    raise MLInputError(
                        "gradient boosting n_estimators must be an integer between "
                        f"1 and {_MAX_BOOSTING_ESTIMATORS}"
                    )
                selected_estimators, validation_samples = _select_gradient_boosting_iterations(
                    estimator, x_train, y_train, self.config.task
                )
                estimator.set_params(n_estimators=selected_estimators, warm_start=False)
                boosting_details = {
                    "temporal_early_stopping": True,
                    "requested_max_estimators": requested_max_estimators,
                    "selected_n_estimators": selected_estimators,
                    "early_stopping_validation_samples": validation_samples,
                    "early_stopping_patience": _BOOSTING_PATIENCE,
                }
            if self.config.algorithm.startswith("knn_"):
                parameter_name = "n_neighbors"
                max_neighbors = int(estimator.get_params(deep=False)[parameter_name])
                if max_neighbors > train_end:
                    estimator.set_params(n_neighbors=train_end)
            steps: list[tuple[str, Any]] = []
            if self.config.algorithm in _SCALED_MODELS:
                steps.append(("scale", StandardScaler()))
            steps.append(("model", estimator))
            pipeline = Pipeline(steps)
            pipeline.fit(x_train, y_train)
            predictions = np.asarray(pipeline.predict(x_test))
        except MLInputError:
            raise
        except (ValueError, TypeError, FloatingPointError) as exc:
            raise MLTrainingError(f"model training failed for {self.config.algorithm}") from exc

        if predictions.shape[0] != len(y_test):
            raise MLTrainingError(
                "estimator returned a prediction count inconsistent with the test set"
            )
        if self.config.task == "regression":
            predicted_numeric = predictions.astype(np.float64)
            actual_numeric = y_test.astype(np.float64)
            if not np.all(np.isfinite(predicted_numeric)):
                raise MLTrainingError("model produced non-finite regression predictions")
            mse = float(mean_squared_error(actual_numeric, predicted_numeric))
            metrics = {
                "mae": float(mean_absolute_error(actual_numeric, predicted_numeric)),
                "rmse": math.sqrt(max(0.0, mse)),
                "r2": float(r2_score(actual_numeric, predicted_numeric)),
            }
        else:
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision_macro": float(
                    precision_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "recall_macro": float(
                    recall_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            }
        if any(not math.isfinite(value) for value in metrics.values()):
            raise MLTrainingError("model evaluation produced non-finite metrics")

        training_fingerprint = _fingerprint(x_train, tuple(y_train.tolist()))
        identity_payload = json.dumps(
            {
                "algorithm": self.config.algorithm,
                "seed": self.config.random_seed,
                "parameters": supplied,
                "data_fingerprint": training_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        resolved_parameters: dict[str, int | float | str | bool | None] = {
            key: value
            for key, value in estimator.get_params(deep=False).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        resolved_parameters["random_seed"] = self.config.random_seed
        resolved_parameters.update(boosting_details)

        self._pipeline = pipeline
        self.metadata = MLModelMetadata(
            model_id=(
                f"{self.config.algorithm}:"
                f"{hashlib.sha256(identity_payload.encode()).hexdigest()[:16]}"
            ),
            algorithm=self.config.algorithm,
            task=self.config.task,
            feature_names=dataset.feature_names,
            training_observations=train_end,
            random_seed=self.config.random_seed,
            library_version=sklearn.__version__,
            data_fingerprint=training_fingerprint,
            parameters=resolved_parameters,
        )
        self.evaluation = MLEvaluation(
            algorithm=self.config.algorithm,
            task=self.config.task,
            training_samples=train_end,
            test_samples=test_size,
            train_end_index=train_end,
            test_start_index=train_end,
            test_end_index=total,
            metrics=metrics,
        )
        return self.evaluation

    def predict(self, rows: Sequence[Sequence[float]]) -> MLPrediction:
        """Predict with a fitted pipeline; candidate code is never generated or executed."""
        if self._pipeline is None or self.metadata is None:
            raise MLPredictionError("model must be fitted before prediction")
        matrix = _as_matrix(rows, width=len(self.metadata.feature_names))
        try:
            raw_predictions = self._pipeline.predict(matrix)
            estimator: Any = self._pipeline.named_steps["model"]
            classes: tuple[float | str, ...] | None = None
            probabilities: tuple[tuple[float, ...], ...] | None = None
            if self.config.task == "classification" and hasattr(estimator, "predict_proba"):
                raw_probabilities = np.asarray(
                    self._pipeline.predict_proba(matrix), dtype=np.float64
                )
                if raw_probabilities.ndim != 2 or not np.all(np.isfinite(raw_probabilities)):
                    raise MLPredictionError("model returned invalid class probabilities")
                class_values = tuple(
                    value.item() if isinstance(value, np.generic) else value
                    for value in estimator.classes_
                )
                classes = class_values
                probabilities = tuple(
                    tuple(float(value) for value in row) for row in raw_probabilities
                )
            values = tuple(
                value.item() if isinstance(value, np.generic) else value
                for value in raw_predictions
            )
            return MLPrediction(
                predictions=values,
                classes=classes,
                probabilities=probabilities,
            )
        except MLPredictionError:
            raise
        except (ValueError, TypeError, FloatingPointError, AttributeError) as exc:
            raise MLPredictionError("prediction failed for the fitted model") from exc


__all__ = ["ClassicalMLModel", "supported_algorithms"]
