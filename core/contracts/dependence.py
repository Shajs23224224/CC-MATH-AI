"""Typed contracts for F23 cross-asset dependence and PCA factors."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

DependenceMethod = Literal[
    "pearson",
    "spearman",
    "kendall",
    "sample_covariance",
    "oas_shrinkage",
]


class DependenceMatrix(BaseModel):
    """Auditable cross-asset covariance or correlation matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    assets: tuple[str, ...] = Field(min_length=2)
    method: DependenceMethod
    kind: Literal["correlation", "covariance"]
    observations: int = Field(ge=3)
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=3)
    matrix: tuple[tuple[float, ...], ...]
    shrinkage_intensity: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_matrix(self) -> DependenceMatrix:
        """Validate dimensions, ordering, units, symmetry and positive semidefiniteness."""
        size = len(self.assets)
        if size < 2 or len(set(self.assets)) != size:
            raise ValueError("assets must contain at least two unique labels")
        if any(not asset or asset != asset.strip().upper() for asset in self.assets):
            raise ValueError("asset labels must be non-empty, trimmed uppercase symbols")
        if tuple(sorted(self.assets)) != self.assets:
            raise ValueError("asset labels must be in deterministic sorted order")
        if self.end_index - self.start_index != self.observations:
            raise ValueError("end_index - start_index must equal observations")
        if len(self.matrix) != size or any(len(row) != size for row in self.matrix):
            raise ValueError("dependence matrix must be square and match the asset count")
        if self.kind == "correlation" and self.method not in {"pearson", "spearman", "kendall"}:
            raise ValueError("correlation matrices require a correlation method")
        if self.kind == "covariance" and self.method not in {"sample_covariance", "oas_shrinkage"}:
            raise ValueError("covariance matrices require a covariance method")
        if (self.method == "oas_shrinkage") != (self.shrinkage_intensity is not None):
            raise ValueError("shrinkage_intensity must be present only for OAS shrinkage")
        values = np.asarray(self.matrix, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("dependence matrix entries must be finite")
        if not np.allclose(values, values.T, rtol=0.0, atol=1e-8):
            raise ValueError("dependence matrix must be symmetric")
        if self.kind == "correlation":
            if np.any(values < -1.0 - 1e-8) or np.any(values > 1.0 + 1e-8):
                raise ValueError("correlation entries must be bounded between -1 and 1")
            if not np.allclose(np.diag(values), 1.0, rtol=0.0, atol=1e-8):
                raise ValueError("correlation matrix diagonal must equal one")
            # Pearson and Spearman matrices from one common complete sample are PSD.
            # Kendall tau-b is pairwise valid but is not guaranteed PSD for finite samples.
            must_be_psd = self.method in {"pearson", "spearman"}
        else:
            if np.any(np.diag(values) < -1e-10):
                raise ValueError("covariance diagonal must be non-negative")
            must_be_psd = True
        if must_be_psd and float(np.min(np.linalg.eigvalsh(values))) < -1e-8:
            raise ValueError("dependence matrix must be positive semidefinite")
        return self


class PCAResult(BaseModel):
    """Principal-component factors, loadings and historical scores."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    assets: tuple[str, ...] = Field(min_length=2)
    observations: int = Field(ge=3)
    standardized: bool
    eigenvalues: tuple[float, ...] = Field(min_length=1)
    explained_variance_ratio: tuple[float, ...] = Field(min_length=1)
    loadings: tuple[tuple[float, ...], ...]
    scores: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def validate_pca(self) -> PCAResult:
        """Check PCA dimensions, spectral ordering and orthonormal loadings."""
        asset_count = len(self.assets)
        component_count = len(self.eigenvalues)
        if len(set(self.assets)) != asset_count or tuple(sorted(self.assets)) != self.assets:
            raise ValueError("PCA assets must be unique and sorted")
        if any(not asset or asset != asset.strip().upper() for asset in self.assets):
            raise ValueError("PCA asset labels must be non-empty uppercase symbols")
        if component_count > asset_count:
            raise ValueError("component count cannot exceed asset count")
        if len(self.explained_variance_ratio) != component_count:
            raise ValueError("explained variance ratios must match component count")
        if len(self.loadings) != asset_count or any(
            len(row) != component_count for row in self.loadings
        ):
            raise ValueError("PCA loadings must have shape (assets, components)")
        if len(self.scores) != self.observations or any(
            len(row) != component_count for row in self.scores
        ):
            raise ValueError("PCA scores must have shape (observations, components)")
        numeric_values = (
            *self.eigenvalues,
            *self.explained_variance_ratio,
            *(value for row in self.loadings for value in row),
            *(value for row in self.scores for value in row),
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("PCA outputs must be finite")
        if any(value < -1e-10 for value in self.eigenvalues):
            raise ValueError("PCA eigenvalues must be non-negative")
        if any(
            self.eigenvalues[index] + 1e-10 < self.eigenvalues[index + 1]
            for index in range(component_count - 1)
        ):
            raise ValueError("PCA eigenvalues must be ordered descending")
        if any(not 0.0 <= ratio <= 1.0 + 1e-8 for ratio in self.explained_variance_ratio):
            raise ValueError("explained variance ratios must be between zero and one")
        if sum(self.explained_variance_ratio) > 1.0 + 1e-7:
            raise ValueError("explained variance ratios cannot sum above one")
        loadings = np.asarray(self.loadings, dtype=float)
        if not np.allclose(loadings.T @ loadings, np.eye(component_count), atol=1e-7, rtol=0.0):
            raise ValueError("PCA loading columns must be orthonormal")
        return self
