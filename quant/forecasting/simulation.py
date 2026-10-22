"""Monte Carlo path generation for probabilistic forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from core.contracts import ReturnPathSimulation

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_MIN_OBSERVATIONS = 20
_MIN_VARIANCE = 1e-14
_MIN_SIMULATIONS = 100


def _prepare_returns(values: Sequence[float] | FloatArray) -> FloatArray:
    """Validate historical simple returns without silently imputing data."""
    try:
        checked = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be a finite numeric sequence") from exc
    if checked.ndim != 1 or checked.size < _MIN_OBSERVATIONS:
        raise ValueError(f"at least {_MIN_OBSERVATIONS} return observations are required")
    if not np.all(np.isfinite(checked)):
        raise ValueError("returns must be finite")
    if np.any(checked <= -1.0):
        raise ValueError("simple returns must be greater than -1")
    if float(np.var(checked, ddof=1)) <= _MIN_VARIANCE:
        raise ValueError("returns must have positive sample variance")
    return checked


def _validate_integer(value: int, name: str, minimum: int) -> int:
    """Validate integer controls while excluding booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _prepare_regimes(
    observations: int,
    regime_labels: Sequence[int] | None,
    transition_matrix: Sequence[Sequence[float]] | None,
) -> tuple[IntArray, FloatArray] | None:
    """Validate aligned training-only labels and a row-stochastic transition matrix."""
    if regime_labels is None and transition_matrix is None:
        return None
    if regime_labels is None or transition_matrix is None:
        raise ValueError("regime_labels and transition_matrix must be supplied together")
    if len(regime_labels) != observations:
        raise ValueError("regime_labels must align exactly with the training returns")

    try:
        labels_raw = tuple(regime_labels)
        if any(
            not isinstance(label, (int, np.integer)) or isinstance(label, bool)
            for label in labels_raw
        ):
            raise ValueError("regime labels must be integer state indices")
        labels = np.asarray(labels_raw, dtype=np.int64)
        matrix = np.asarray(transition_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("regime inputs must be numeric and rectangular") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    state_count = int(matrix.shape[0])
    if not 2 <= state_count <= 6:
        raise ValueError("transition_matrix must contain between 2 and 6 states")
    if labels.ndim != 1 or labels.size != observations:
        raise ValueError("regime_labels must be a one-dimensional sequence")
    if np.any(labels < 0) or np.any(labels >= state_count):
        raise ValueError("regime labels must be valid state indices")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("transition probabilities must be finite and lie in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("each transition matrix row must sum to one")
    for state in range(state_count):
        if int(np.count_nonzero(labels == state)) < 2:
            raise ValueError("each regime must contain at least two training observations")
    return labels, matrix


def _simulate_paths_array(
    returns: Sequence[float] | FloatArray,
    *,
    horizon: int,
    simulations: int,
    random_seed: int,
    regime_labels: Sequence[int] | None = None,
    transition_matrix: Sequence[Sequence[float]] | None = None,
) -> tuple[FloatArray, int, IntArray | None, FloatArray | None]:
    """Internal generator used by both the public simulation API and forecasting."""
    checked = _prepare_returns(returns)
    steps = _validate_integer(horizon, "horizon", 1)
    draws = _validate_integer(simulations, "simulations", _MIN_SIMULATIONS)
    seed = _validate_integer(random_seed, "random_seed", 0)
    regimes = _prepare_regimes(len(checked), regime_labels, transition_matrix)
    rng = np.random.default_rng(seed)
    paths = np.empty((draws, steps), dtype=np.float64)

    if regimes is None:
        sampled_indices = rng.integers(0, checked.size, size=(draws, steps))
        paths[:, :] = checked[sampled_indices]
        return paths, seed, None, None

    labels, matrix = regimes
    state_count = int(matrix.shape[0])
    pools = tuple(checked[labels == state] for state in range(state_count))
    previous_states = np.full(draws, int(labels[-1]), dtype=np.int64)

    for step in range(steps):
        next_states = np.empty(draws, dtype=np.int64)
        for state in range(state_count):
            mask = previous_states == state
            count = int(np.count_nonzero(mask))
            if count:
                next_states[mask] = rng.choice(state_count, size=count, p=matrix[state])
        for state in range(state_count):
            mask = next_states == state
            count = int(np.count_nonzero(mask))
            if count:
                sampled = rng.integers(0, len(pools[state]), size=count)
                paths[mask, step] = pools[state][sampled]
        previous_states = next_states

    return paths, seed, labels, matrix


def simulate_return_paths(
    returns: Sequence[float] | FloatArray,
    *,
    horizon: int,
    simulations: int = 10_000,
    random_seed: int = 27,
    regime_labels: Sequence[int] | None = None,
    transition_matrix: Sequence[Sequence[float]] | None = None,
) -> ReturnPathSimulation:
    """Simulate future simple-return paths by IID or Markov regime-conditioned bootstrap.

    All inputs must belong to the historical training window. When regime metadata is
    provided, each future state follows the supplied transition matrix and each return
    is sampled from the historical observations assigned to that state. The transition
    matrix is expected to have been estimated on training data only.
    """
    paths, seed, labels, matrix = _simulate_paths_array(
        returns,
        horizon=horizon,
        simulations=simulations,
        random_seed=random_seed,
        regime_labels=regime_labels,
        transition_matrix=transition_matrix,
    )
    checked = _prepare_returns(returns)
    if matrix is None or labels is None:
        return ReturnPathSimulation(
            model="iid_empirical_bootstrap",
            observations=len(checked),
            horizon=horizon,
            simulations=simulations,
            random_seed=seed,
            simulated_returns=tuple(tuple(float(value) for value in row) for row in paths),
            methodology=(
                "IID empirical bootstrap with replacement from training-window simple returns; "
                "future return dependence is not modelled."
            ),
        )
    return ReturnPathSimulation(
        model="markov_regime_bootstrap",
        observations=len(checked),
        horizon=horizon,
        simulations=simulations,
        random_seed=seed,
        simulated_returns=tuple(tuple(float(value) for value in row) for row in paths),
        current_regime=int(labels[-1]),
        state_count=int(matrix.shape[0]),
        transition_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        methodology=(
            "Markov regime bootstrap using training-window state labels, the supplied "
            "row-stochastic transition matrix, and within-state empirical returns."
        ),
    )


__all__ = ["simulate_return_paths"]
