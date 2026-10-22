"""Bounded polynomial symbolic-regression search using explicit numerical primitives."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from math import comb

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from core.contracts.ml import MLDataset, SymbolicRegressionResult

from .errors import MLInputError


def _matrix(rows: Sequence[Sequence[float]], width: int) -> np.ndarray:
    try:
        matrix = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MLInputError("symbolic-regression features must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != width or not np.all(np.isfinite(matrix)):
        raise MLInputError("symbolic-regression features are ragged or non-finite")
    return matrix


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    errors = predicted - actual
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    variance = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 0.0 if variance == 0.0 else float(1.0 - np.sum(errors**2) / variance)
    if not all(math.isfinite(value) for value in (rmse, mae, r2)):
        raise MLInputError("symbolic-regression evaluation produced non-finite metrics")
    return rmse, mae, r2


def _expression(intercept: float, terms: Sequence[str], coefficients: Sequence[float]) -> str:
    pieces = [f"{intercept:.10g}"]
    scale = max(1.0, abs(intercept), *(abs(value) for value in coefficients))
    for name, coefficient in zip(terms, coefficients, strict=True):
        if abs(coefficient) > 1e-9 * scale:
            pieces.append(f"({coefficient:+.10g} * {name})")
    return "y ≈ " + " ".join(pieces)


def discover_polynomial_relation(
    dataset: MLDataset,
    *,
    max_degree: int = 3,
    max_terms: int = 128,
    complexity_penalty: float = 1e-5,
) -> SymbolicRegressionResult:
    """Search a bounded polynomial grammar and select degree on validation only.

    The data must already be ordered chronologically. The last 20% is held out as a
    final test set; degree selection uses only an earlier validation segment. Coefficients
    are fitted on the non-test prefix after degree selection. The returned expression is
    an experimental candidate, not a mathematical proof of the generating law.
    """
    if dataset.task != "regression":
        raise MLInputError("symbolic relation discovery requires a regression dataset")
    if isinstance(max_degree, bool) or not 1 <= max_degree <= 4:
        raise MLInputError("max_degree must be an integer between 1 and 4")
    if isinstance(max_terms, bool) or not 1 <= max_terms <= 512:
        raise MLInputError("max_terms must be an integer between 1 and 512")
    if not math.isfinite(complexity_penalty) or complexity_penalty < 0.0:
        raise MLInputError("complexity_penalty must be finite and non-negative")
    features = _matrix(dataset.features, len(dataset.feature_names))
    target = np.asarray(dataset.target, dtype=np.float64)
    total, width = features.shape
    term_bound = comb(width + max_degree, max_degree) - 1
    if term_bound > max_terms:
        raise MLInputError(
            f"polynomial grammar would create {term_bound} terms; max_terms is {max_terms}"
        )
    test_size = max(3, int(math.ceil(total * 0.2)))
    development_end = total - test_size
    validation_size = max(2, int(math.ceil(development_end * 0.2)))
    train_end = development_end - validation_size
    if train_end < max(6, width + 2):
        raise MLInputError(
            "dataset is too small for chronological train/validation/test partitions"
        )
    x_train, y_train = features[:train_end], target[:train_end]
    x_validation = features[train_end:development_end]
    y_validation = target[train_end:development_end]
    x_development, y_development = features[:development_end], target[:development_end]
    x_test, y_test = features[development_end:], target[development_end:]

    best: tuple[float, int] | None = None
    for degree in range(1, max_degree + 1):
        basis = PolynomialFeatures(degree=degree, include_bias=False)
        try:
            x_train_basis = basis.fit_transform(x_train)
            x_validation_basis = basis.transform(x_validation)
            if not np.all(np.isfinite(x_train_basis)) or not np.all(
                np.isfinite(x_validation_basis)
            ):
                raise MLInputError("polynomial expansion overflowed; rescale features")
            estimator = LinearRegression()
            estimator.fit(x_train_basis, y_train)
            validation_predictions = estimator.predict(x_validation_basis)
            rmse = float(np.sqrt(np.mean((validation_predictions - y_validation) ** 2)))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise MLInputError(f"polynomial candidate of degree {degree} failed") from exc
        term_count = int(x_train_basis.shape[1])
        score = rmse + complexity_penalty * term_count
        if not math.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, degree)
    if best is None:
        raise MLInputError("no finite polynomial candidate was found")

    _, selected_degree = best
    selected_basis = PolynomialFeatures(degree=selected_degree, include_bias=False)
    x_development_basis = selected_basis.fit_transform(x_development)
    x_test_basis = selected_basis.transform(x_test)
    if not np.all(np.isfinite(x_development_basis)) or not np.all(np.isfinite(x_test_basis)):
        raise MLInputError("selected polynomial expansion overflowed; rescale features")
    final_estimator = LinearRegression()
    final_estimator.fit(x_development_basis, y_development)
    test_predictions = final_estimator.predict(x_test_basis)
    test_rmse, test_mae, test_r2 = _metrics(y_test, test_predictions)
    # Validation error is retained for the selected degree before the final refit.
    selected_basis_for_validation = PolynomialFeatures(degree=selected_degree, include_bias=False)
    selected_train_basis = selected_basis_for_validation.fit_transform(x_train)
    selected_validation_basis = selected_basis_for_validation.transform(x_validation)
    validation_model = LinearRegression().fit(selected_train_basis, y_train)
    validation_rmse = float(
        np.sqrt(np.mean((validation_model.predict(selected_validation_basis) - y_validation) ** 2))
    )
    if not math.isfinite(validation_rmse):
        raise MLInputError("selected model produced non-finite validation metrics")

    terms = tuple(selected_basis.get_feature_names_out(dataset.feature_names).tolist())
    coefficients = tuple(float(value) for value in final_estimator.coef_)
    intercept = float(final_estimator.intercept_)
    formula = _expression(intercept, terms, coefficients)
    signature = f"{dataset.dataset_id}|{selected_degree}|{formula}|{train_end}|{development_end}"
    return SymbolicRegressionResult(
        candidate_id=hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24],
        dataset_id=dataset.dataset_id,
        feature_names=dataset.feature_names,
        degree=selected_degree,
        basis_terms=terms,
        coefficients=coefficients,
        intercept=intercept,
        expression=formula,
        complexity=1 + sum(abs(value) > 1e-9 for value in coefficients),
        training_samples=train_end,
        validation_samples=validation_size,
        test_samples=test_size,
        train_end_index=train_end,
        validation_end_index=development_end,
        test_start_index=development_end,
        validation_rmse=validation_rmse,
        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2,
        max_degree=max_degree,
        max_terms=max_terms,
        limitations=(
            "Search space is limited to polynomial terms of the configured maximum degree.",
            "The test tail is evaluated once and is not used to select the degree.",
            "A low test error does not prove uniqueness or mathematical truth of the expression.",
            "No generated source code is executed; evaluation uses authorized "
            "scikit-learn primitives.",
        ),
    )


def predict_polynomial_relation(
    candidate: SymbolicRegressionResult,
    rows: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Evaluate a stored polynomial representation without eval, exec or dynamic code."""
    features = _matrix(rows, len(candidate.feature_names))
    basis = PolynomialFeatures(degree=candidate.degree, include_bias=False)
    expanded = basis.fit_transform(features)
    if not np.all(np.isfinite(expanded)):
        raise MLInputError("polynomial expansion overflowed; rescale features")
    if expanded.shape[1] != len(candidate.coefficients):
        raise MLInputError("stored polynomial representation is incompatible with the input")
    predictions = expanded @ np.asarray(candidate.coefficients) + candidate.intercept
    if not np.all(np.isfinite(predictions)):
        raise MLInputError("polynomial candidate produced non-finite output")
    return tuple(float(value) for value in predictions)


__all__ = ["discover_polynomial_relation", "predict_polynomial_relation"]
