"""Exploratory feature/target association discovery without causal claims."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import spearmanr

from core.contracts.ml import MLDataset, PatternRecord
from .errors import MLInputError


def discover_patterns(
    dataset: MLDataset,
    *,
    min_abs_correlation: float = 0.7,
) -> tuple[PatternRecord, ...]:
    """Record strong Pearson associations, rank correlations and half-sample stability.

    This is a screening heuristic, not causal inference and not proof of predictive
    generalization. A feature is reported only when both half-window correlations can
    be estimated and its full-sample absolute Pearson correlation meets the threshold.
    """
    if dataset.task != "regression":
        raise MLInputError("association discovery currently requires a numeric regression target")
    if not np.isfinite(min_abs_correlation) or not 0.0 <= min_abs_correlation <= 1.0:
        raise MLInputError("min_abs_correlation must lie in [0, 1]")
    features = np.asarray(dataset.features, dtype=np.float64)
    target = np.asarray(dataset.target, dtype=np.float64)
    half = len(target) // 2
    if half < 3 or len(target) - half < 3:
        raise MLInputError("pattern discovery requires at least three observations per half-window")

    records: list[PatternRecord] = []
    for index, name in enumerate(dataset.feature_names):
        values = features[:, index]
        if np.ptp(values) == 0.0 or np.ptp(target) == 0.0:
            continue
        pearson = float(np.corrcoef(values, target)[0, 1])
        spearman = float(spearmanr(values, target).statistic)
        first = float(np.corrcoef(values[:half], target[:half])[0, 1])
        second = float(np.corrcoef(values[half:], target[half:])[0, 1])
        if not all(np.isfinite(value) for value in (pearson, spearman, first, second)):
            continue
        if abs(pearson) < min_abs_correlation:
            continue
        stability = abs(first - second)
        signature = f"{dataset.dataset_id}|{name}|{pearson:.12g}|{spearman:.12g}"
        records.append(
            PatternRecord(
                pattern_id=hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24],
                dataset_id=dataset.dataset_id,
                feature_name=name,
                target_name=dataset.target_name,
                observations=len(target),
                pearson_correlation=pearson,
                spearman_correlation=spearman,
                stability_delta=stability,
                hypothesis=(
                    f"{name} is associated with {dataset.target_name} in the observed sample; "
                    "the association should be re-evaluated on independent data."
                ),
                limitations=(
                    "Correlation does not imply causation.",
                    "Thresholded discovery is exploratory and can miss nonlinear or weak relationships.",
                    "Half-window stability is a diagnostic, not a significance test.",
                ),
            )
        )
    return tuple(sorted(records, key=lambda item: abs(item.pearson_correlation), reverse=True))


__all__ = ["discover_patterns"]
