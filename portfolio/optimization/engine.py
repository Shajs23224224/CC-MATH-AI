"""Deterministic long-only portfolio optimization for F36."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from core.contracts import SignalType, TargetAllocation

from .contracts import (
    AssetExclusion,
    ConstraintDiagnostic,
    PortfolioAssetInput,
    PortfolioOptimizationConfig,
    PortfolioOptimizationRequest,
    PortfolioOptimizationResult,
)


def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    config: PortfolioOptimizationConfig | None = None,
) -> PortfolioOptimizationResult:
    """Create a reproducible, constrained target allocation from F35 decisions.

    The solver is long-only and fully invested. F35 abstentions, SELL signals,
    caller-ineligible assets, and HOLD decisions without an existing position are
    excluded. An existing HOLD may be retained or reduced, but it cannot be increased.
    No order is created and no Risk Governor approval is implied.
    """
    policy = config or PortfolioOptimizationConfig()
    fingerprint = _fingerprint(request, policy)
    original_weights = np.asarray(
        [asset.current_weight for asset in request.assets], dtype=np.float64
    )
    covariance_original = np.asarray(request.covariance.matrix, dtype=np.float64)
    covariance, ridge = _regularize_covariance(covariance_original, policy.covariance_ridge)
    expected_returns_original = np.asarray(
        [asset.decision.record.expected_return for asset in request.assets], dtype=np.float64
    )

    exclusions: list[AssetExclusion] = []
    active_indices: list[int] = []
    for index, asset in enumerate(request.assets):
        reason = _exclusion_reason(asset)
        if reason is not None:
            exclusions.append(AssetExclusion(asset_symbol=asset.asset_symbol, reason=reason))
        else:
            active_indices.append(index)

    if policy.method == "black_litterman" and not request.view_returns:
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Black-Litterman requires at least one explicit view.",
            status="FAILED",
        )
    if policy.method != "black_litterman" and request.view_returns:
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Black-Litterman views are only accepted with method='black_litterman'.",
            status="FAILED",
        )

    if not active_indices:
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "No active asset remains after F35 and eligibility gates.",
            status="INFEASIBLE",
        )

    if policy.sector_limits and any(
        request.assets[index].sector is None for index in active_indices
    ):
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Sector labels are required for every active asset when sector limits are configured.",
            status="FAILED",
        )

    lower = np.zeros(len(request.assets), dtype=np.float64)
    upper = np.zeros(len(request.assets), dtype=np.float64)
    for index, asset in enumerate(request.assets):
        if index not in active_indices:
            if asset.minimum_weight > 0.0:
                return _failure(
                    request,
                    policy,
                    fingerprint,
                    exclusions,
                    ridge,
                    f"Excluded asset {asset.asset_symbol} has a positive minimum weight.",
                    status="INFEASIBLE",
                )
            continue
        lower[index] = asset.minimum_weight
        asset_cap = asset.maximum_weight
        if asset.decision.record.signal == SignalType.HOLD:
            asset_cap = min(asset_cap, asset.current_weight)
        upper[index] = asset_cap

    if math.fsum(lower) > 1.0 + 1e-8 or math.fsum(upper) < 1.0 - 1e-8:
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Per-asset minimum/maximum bounds cannot sum to a fully invested portfolio.",
            status="INFEASIBLE",
        )

    for index, asset in enumerate(request.assets):
        if index in active_indices and lower[index] > upper[index] + 1e-12:
            return _failure(
                request,
                policy,
                fingerprint,
                exclusions,
                ridge,
                f"Effective bounds for {asset.asset_symbol} are infeasible.",
                status="INFEASIBLE",
            )

    for limit in policy.sector_limits:
        members = [
            index
            for index in active_indices
            if request.assets[index].sector == limit.sector
        ]
        minimum_sector_weight = math.fsum(lower[index] for index in members)
        if minimum_sector_weight > limit.maximum_weight + 1e-8:
            return _failure(
                request,
                policy,
                fingerprint,
                exclusions,
                ridge,
                f"Minimum weights breach sector cap {limit.sector}.",
                status="INFEASIBLE",
            )

    if policy.max_concentration_hhi is not None:
        theoretical_minimum_hhi = 1.0 / len(active_indices)
        if policy.max_concentration_hhi + 1e-8 < theoretical_minimum_hhi:
            return _failure(
                request,
                policy,
                fingerprint,
                exclusions,
                ridge,
                "Maximum HHI is below the equal-weight lower bound for active assets.",
                status="INFEASIBLE",
            )

    if policy.method == "black_litterman":
        try:
            effective_returns = _black_litterman_returns(covariance, request, policy)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return _failure(
                request,
                policy,
                fingerprint,
                exclusions,
                ridge,
                f"Black-Litterman posterior could not be computed: {exc}",
                status="FAILED",
            )
    else:
        effective_returns = expected_returns_original.copy()

    active_covariance = covariance[np.ix_(active_indices, active_indices)]
    active_returns = effective_returns[active_indices]
    active_lower = lower[active_indices]
    active_upper = upper[active_indices]

    target = _base_target(policy.method, active_covariance)
    if policy.method == "black_litterman":
        objective_method = "mean_variance"
    else:
        objective_method = policy.method

    active_bounds = list(zip(active_lower.tolist(), active_upper.tolist(), strict=True))
    constraints = _scipy_constraints(
        request=request,
        policy=policy,
        active_indices=active_indices,
        original_weights=original_weights,
    )
    initial = _bounded_simplex_start(active_lower, active_upper)
    feasibility = minimize(
        fun=lambda weights: float(np.sum(np.square(weights - initial))),
        x0=initial,
        method="SLSQP",
        bounds=active_bounds,
        constraints=constraints,
        options={
            "maxiter": policy.max_iterations,
            "ftol": policy.solver_tolerance,
            "disp": False,
        },
    )

    feasibility_weights = np.asarray(feasibility.x, dtype=np.float64)
    if not np.all(np.isfinite(feasibility_weights)):
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Feasibility solver returned non-finite weights.",
            status="FAILED",
            iterations=int(getattr(feasibility, "nit", 0)),
        )
    feasibility_diagnostics = _diagnostics(
        full_weights=_expand_weights(feasibility_weights, active_indices, len(request.assets)),
        request=request,
        policy=policy,
        lower=lower,
        upper=upper,
        original_weights=original_weights,
    )
    if not _all_constraints_satisfied(feasibility_diagnostics):
        message = (
            "The constrained solver could not find a feasible starting allocation; "
            f"feasibility status: {feasibility.message}"
        )
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            message,
            status="FAILED",
            iterations=int(getattr(feasibility, "nit", 0)),
            constraints=feasibility_diagnostics,
        )

    def objective(weights: np.ndarray) -> float:
        """Evaluate one deterministic objective and its explicit trading regularizers."""
        weights = np.asarray(weights, dtype=np.float64)
        variance = max(float(weights @ active_covariance @ weights), 1e-16)
        portfolio_volatility = math.sqrt(variance)
        expected_return = float(active_returns @ weights)
        full_weights = _expand_weights(weights, active_indices, len(request.assets))
        turnover, cost_fraction = _trading_costs(
            full_weights, original_weights, policy.transaction_cost_bps
        )
        turnover_regularizer = policy.turnover_penalty * turnover

        if objective_method == "equal_weight":
            value = float(np.sum(np.square(weights - target)))
            return value + turnover_regularizer
        if objective_method == "inverse_volatility":
            value = float(np.sum(np.square(weights - target)))
            return value + turnover_regularizer
        if objective_method == "hrp":
            value = float(np.sum(np.square(weights - target)))
            return value + turnover_regularizer
        if objective_method == "minimum_variance":
            return 0.5 * variance + turnover_regularizer
        if objective_method == "mean_variance":
            net_return = expected_return - cost_fraction
            return 0.5 * policy.risk_aversion * variance - net_return + turnover_regularizer
        if objective_method == "maximum_sharpe":
            net_excess_return = (
                expected_return
                - policy.risk_free_rate
                - cost_fraction
                - turnover_regularizer
            )
            return -net_excess_return / portfolio_volatility
        if objective_method == "risk_parity":
            component_risk = weights * (active_covariance @ weights)
            risk_shares = component_risk / variance
            equal_share_loss = float(np.sum(np.square(risk_shares - 1.0 / len(weights))))
            return equal_share_loss + turnover_regularizer
        if objective_method == "maximum_diversification":
            weighted_volatility = float(weights @ np.sqrt(np.diag(active_covariance)))
            ratio = weighted_volatility / portfolio_volatility
            return -ratio + turnover_regularizer
        raise ValueError(f"unsupported objective method: {objective_method}")

    result = minimize(
        fun=objective,
        x0=feasibility_weights,
        method="SLSQP",
        bounds=active_bounds,
        constraints=constraints,
        options={
            "maxiter": policy.max_iterations,
            "ftol": policy.solver_tolerance,
            "disp": False,
        },
    )
    active_solution = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(active_solution)):
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Optimizer returned non-finite weights.",
            status="FAILED",
            iterations=int(getattr(result, "nit", 0)),
        )
    full_weights = _expand_weights(active_solution, active_indices, len(request.assets))
    diagnostics = _diagnostics(
        full_weights=full_weights,
        request=request,
        policy=policy,
        lower=lower,
        upper=upper,
        original_weights=original_weights,
    )
    if not np.all(np.isfinite(full_weights)):
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Optimizer returned non-finite target weights.",
            status="FAILED",
            iterations=int(getattr(result, "nit", 0)),
            constraints=diagnostics,
        )

    if not _all_constraints_satisfied(diagnostics):
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            f"Optimizer returned a constraint-breaching candidate: {result.message}",
            status="FAILED",
            iterations=int(getattr(result, "nit", 0)),
            constraints=diagnostics,
        )

    allocation_rows = tuple(
        TargetAllocation(
            asset_symbol=asset.asset_symbol,
            target_weight=float(max(0.0, min(1.0, full_weights[index]))),
        )
        for index, asset in enumerate(request.assets)
    )
    turnover, cost_fraction = _trading_costs(
        full_weights, original_weights, policy.transaction_cost_bps
    )
    portfolio_return = float(effective_returns @ full_weights)
    raw_variance = float(full_weights @ covariance @ full_weights)
    if not all(
        math.isfinite(value)
        for value in (portfolio_return, raw_variance, turnover, cost_fraction, float(result.fun))
    ):
        return _failure(
            request,
            policy,
            fingerprint,
            exclusions,
            ridge,
            "Optimizer diagnostics contain non-finite return, risk, cost, turnover or objective values.",
            status="FAILED",
            iterations=int(getattr(result, "nit", 0)),
            constraints=diagnostics,
        )
    variance = max(raw_variance, 0.0)
    volatility = math.sqrt(variance)
    net_excess = portfolio_return - cost_fraction - policy.risk_free_rate
    sharpe = net_excess / volatility if volatility > 1e-12 else None
    concentration = float(full_weights @ full_weights)
    status = "OPTIMAL" if bool(result.success) else "FEASIBLE"
    message = str(result.message)[:2000] or (
        "Converged." if result.success else "Feasible solution found."
    )
    return PortfolioOptimizationResult(
        status=status,
        method=policy.method,
        config_id=policy.config_id,
        config_version=policy.version,
        asset_symbols=tuple(asset.asset_symbol for asset in request.assets),
        allocations=allocation_rows,
        exclusions=tuple(sorted(exclusions, key=lambda item: item.asset_symbol)),
        constraints=diagnostics,
        objective_value=float(result.fun) if math.isfinite(float(result.fun)) else None,
        expected_return=portfolio_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        turnover=turnover,
        estimated_transaction_cost_fraction=cost_fraction,
        concentration_hhi=concentration,
        covariance_regularization=ridge,
        solver_success=bool(result.success),
        solver_message=message,
        iterations=int(getattr(result, "nit", 0)) + int(getattr(feasibility, "nit", 0)),
        as_of=request.as_of,
        horizon=request.horizon,
        data_snapshot_id=request.data_snapshot_id,
        covariance_data_snapshot_id=request.covariance_data_snapshot_id,
        decision_fingerprints=tuple(
            asset.decision.decision_fingerprint for asset in request.assets
        ),
        optimization_fingerprint=fingerprint,
    )


def _exclusion_reason(asset: PortfolioAssetInput) -> str | None:
    """Return the first deterministic reason an asset is not eligible for allocation."""
    if not asset.eligible:
        return "caller marked asset ineligible"
    decision = asset.decision
    if decision.status != "DECIDED" or decision.abstained:
        return "F35 decision abstained; allocation fails closed"
    if decision.record.signal == SignalType.SELL:
        return "F35 signal is SELL; long-only allocation excludes the asset"
    if decision.record.signal == SignalType.HOLD and asset.current_weight <= 1e-12:
        return "F35 signal is HOLD and no existing position is available to retain"
    return None


def _regularize_covariance(
    covariance: np.ndarray,
    relative_floor: float,
) -> tuple[np.ndarray, float]:
    """Symmetrize and apply a small, recorded ridge only when needed."""
    symmetric = (covariance + covariance.T) / 2.0
    scale = max(float(np.mean(np.diag(symmetric))), 1e-12)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    floor = max(relative_floor, 1e-12) * scale
    ridge = max(0.0, floor - float(np.min(eigenvalues)))
    regularized = symmetric + np.eye(symmetric.shape[0], dtype=np.float64) * ridge
    return regularized, ridge


def _base_target(method: str, covariance: np.ndarray) -> np.ndarray:
    """Return an unconstrained target for heuristic methods."""
    count = covariance.shape[0]
    if count == 1:
        return np.ones(1, dtype=np.float64)
    if method == "equal_weight":
        return np.full(count, 1.0 / count, dtype=np.float64)
    if method == "inverse_volatility":
        inverse = 1.0 / np.sqrt(np.maximum(np.diag(covariance), 1e-16))
        return inverse / np.sum(inverse)
    if method == "hrp":
        return _hrp_weights(covariance)
    return np.full(count, 1.0 / count, dtype=np.float64)


def _hrp_weights(covariance: np.ndarray) -> np.ndarray:
    """Compute Hierarchical Risk Parity weights using deterministic clustering."""
    count = covariance.shape[0]
    if count == 1:
        return np.ones(1, dtype=np.float64)
    if count == 2:
        inverse = 1.0 / np.maximum(np.diag(covariance), 1e-16)
        return inverse / np.sum(inverse)

    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-16))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.sqrt(np.maximum((1.0 - correlation) / 2.0, 0.0))
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="single", optimal_ordering=True)
    ordered = [int(index) for index in leaves_list(tree)]
    weights = np.ones(count, dtype=np.float64)
    clusters = [ordered]
    while clusters:
        next_clusters: list[list[int]] = []
        split_occurred = False
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split_occurred = True
            midpoint = len(cluster) // 2
            left, right = cluster[:midpoint], cluster[midpoint:]
            left_variance = _inverse_variance_cluster(covariance, left)
            right_variance = _inverse_variance_cluster(covariance, right)
            denominator = left_variance + right_variance
            left_fraction = 0.5 if denominator <= 1e-16 else 1.0 - left_variance / denominator
            weights[left] *= left_fraction
            weights[right] *= 1.0 - left_fraction
            next_clusters.extend((left, right))
        if not split_occurred:
            break
        clusters = next_clusters
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("HRP clustering produced invalid weights")
    return weights / total


def _inverse_variance_cluster(covariance: np.ndarray, cluster: list[int]) -> float:
    """Estimate a cluster's variance using inverse-variance weights within the cluster."""
    subset = covariance[np.ix_(cluster, cluster)]
    inverse = 1.0 / np.maximum(np.diag(subset), 1e-16)
    weights = inverse / np.sum(inverse)
    return max(float(weights @ subset @ weights), 1e-16)


def _black_litterman_returns(
    covariance: np.ndarray,
    request: PortfolioOptimizationRequest,
    config: PortfolioOptimizationConfig,
) -> np.ndarray:
    """Calculate posterior returns from equilibrium prior and confidence-weighted views."""
    market_weights = (
        np.asarray(request.equilibrium_weights, dtype=np.float64)
        if request.equilibrium_weights
        else np.full(covariance.shape[0], 1.0 / covariance.shape[0], dtype=np.float64)
    )
    prior = config.risk_aversion * covariance @ market_weights
    view_matrix = np.asarray(request.view_matrix, dtype=np.float64)
    view_returns = np.asarray(request.view_returns, dtype=np.float64)
    confidences = np.asarray(request.view_confidences, dtype=np.float64)
    tau_covariance = config.black_litterman_tau * covariance
    projected = view_matrix @ tau_covariance @ view_matrix.T
    uncertainty = np.maximum(np.diag(projected), 1e-12)
    confidence_adjustment = (1.0 - confidences) / confidences
    omega = np.diag(np.maximum(uncertainty * confidence_adjustment, 1e-12))
    system = projected + omega
    posterior = prior + tau_covariance @ view_matrix.T @ np.linalg.pinv(system) @ (
        view_returns - view_matrix @ prior
    )
    if not np.all(np.isfinite(posterior)):
        raise ValueError("Black-Litterman posterior produced non-finite expected returns")
    total_expected_returns = posterior + config.risk_free_rate
    if not np.all(np.isfinite(total_expected_returns)):
        raise ValueError("Black-Litterman total expected returns are non-finite")
    return total_expected_returns


def _bounded_simplex_start(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Construct a deterministic fully invested point inside per-asset bounds."""
    weights = np.clip(np.full(len(lower), 1.0 / len(lower)), lower, upper)
    for _ in range(len(lower) + 2):
        residual = 1.0 - float(np.sum(weights))
        if abs(residual) <= 1e-12:
            break
        capacity = upper - weights if residual > 0.0 else weights - lower
        eligible = capacity > 1e-12
        if not np.any(eligible):
            break
        step = min(abs(residual) / float(np.sum(capacity[eligible])), 1.0)
        weights[eligible] += math.copysign(step, residual) * capacity[eligible]
        weights = np.minimum(upper, np.maximum(lower, weights))
    return weights


def _expand_weights(
    active_weights: np.ndarray,
    active_indices: list[int],
    total_count: int,
) -> np.ndarray:
    """Map active solver weights back to stable request order with excluded weights at zero."""
    full = np.zeros(total_count, dtype=np.float64)
    for local_index, source_index in enumerate(active_indices):
        full[source_index] = active_weights[local_index]
    return full


def _trading_costs(
    target_weights: np.ndarray,
    current_weights: np.ndarray,
    cost_bps: float,
) -> tuple[float, float]:
    """Return one-way turnover and estimated full-turn trading cost fraction.

    An implicit cash balance completes the current portfolio to 100%; deploying cash
    therefore counts as turnover. Commissions apply to absolute changes in risky-asset
    notional only, not to the cash balance itself.
    """
    current_cash = max(0.0, 1.0 - float(np.sum(current_weights)))
    asset_traded_notional = float(np.sum(np.abs(target_weights - current_weights)))
    turnover = 0.5 * (asset_traded_notional + current_cash)
    cost_fraction = (cost_bps / 10_000.0) * asset_traded_notional
    return turnover, cost_fraction


def _scipy_constraints(
    request: PortfolioOptimizationRequest,
    policy: PortfolioOptimizationConfig,
    active_indices: list[int],
    original_weights: np.ndarray,
) -> list[dict[str, Any]]:
    """Translate simplex, turnover, concentration and sector limits to SLSQP."""
    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    ]
    if policy.max_turnover is not None:
        current_cash = max(0.0, 1.0 - float(np.sum(original_weights)))
        turnover_limit = policy.max_turnover

        def turnover_slack(weights: np.ndarray) -> float:
            full = _expand_weights(weights, active_indices, len(request.assets))
            traded = float(np.sum(np.abs(full - original_weights))) + current_cash
            return turnover_limit - 0.5 * traded

        constraints.append({"type": "ineq", "fun": turnover_slack})

    if policy.max_concentration_hhi is not None:
        concentration_limit = policy.max_concentration_hhi
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: concentration_limit - float(weights @ weights),
            }
        )

    configured_limits = {limit.sector: limit.maximum_weight for limit in policy.sector_limits}
    indices_by_sector: dict[str, list[int]] = defaultdict(list)
    for local_index, full_index in enumerate(active_indices):
        sector = request.assets[full_index].sector
        if sector is not None and sector in configured_limits:
            indices_by_sector[sector].append(local_index)
    for sector, local_indices in sorted(indices_by_sector.items()):
        cap = configured_limits[sector]
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, indices=tuple(local_indices), maximum=cap: maximum
                - float(np.sum(weights[list(indices)])),
            }
        )
    return constraints


def _diagnostics(
    full_weights: np.ndarray,
    request: PortfolioOptimizationRequest,
    policy: PortfolioOptimizationConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    original_weights: np.ndarray,
) -> tuple[ConstraintDiagnostic, ...]:
    """Measure every explicit hard constraint against a shared numerical tolerance."""
    tolerance = 5e-6
    diagnostics: list[ConstraintDiagnostic] = []
    total = float(np.sum(full_weights))
    diagnostics.append(
        ConstraintDiagnostic(
            constraint_id="fully_invested",
            satisfied=abs(total - 1.0) <= tolerance,
            measured=total,
            limit=1.0,
            detail="Long-only target weights must sum to one.",
        )
    )
    for index, asset in enumerate(request.assets):
        weight = float(full_weights[index])
        diagnostics.append(
            ConstraintDiagnostic(
                constraint_id=f"minimum_weight:{asset.asset_symbol}",
                satisfied=weight >= lower[index] - tolerance,
                measured=weight,
                limit=float(lower[index]),
                detail="Target weight must be at or above its effective minimum.",
            )
        )
        diagnostics.append(
            ConstraintDiagnostic(
                constraint_id=f"maximum_weight:{asset.asset_symbol}",
                satisfied=weight <= upper[index] + tolerance,
                measured=weight,
                limit=float(upper[index]),
                detail="Target weight must be at or below its effective maximum.",
            )
        )

    for limit in policy.sector_limits:
        weight = math.fsum(
            float(full_weights[index])
            for index, asset in enumerate(request.assets)
            if asset.sector == limit.sector
        )
        diagnostics.append(
            ConstraintDiagnostic(
                constraint_id=f"sector_limit:{limit.sector}",
                satisfied=weight <= limit.maximum_weight + tolerance,
                measured=weight,
                limit=limit.maximum_weight,
                detail="Aggregate target weight for the configured sector.",
            )
        )

    turnover, _ = _trading_costs(
        full_weights,
        original_weights,
        policy.transaction_cost_bps,
    )
    if policy.max_turnover is not None:
        diagnostics.append(
            ConstraintDiagnostic(
                constraint_id="maximum_turnover",
                satisfied=turnover <= policy.max_turnover + tolerance,
                measured=turnover,
                limit=policy.max_turnover,
                detail="One-way turnover including implicit cash deployment.",
            )
        )

    concentration = float(full_weights @ full_weights)
    if policy.max_concentration_hhi is not None:
        diagnostics.append(
            ConstraintDiagnostic(
                constraint_id="maximum_concentration_hhi",
                satisfied=concentration <= policy.max_concentration_hhi + tolerance,
                measured=concentration,
                limit=policy.max_concentration_hhi,
                detail="Herfindahl-Hirschman index of target weights.",
            )
        )
    return tuple(diagnostics)


def _all_constraints_satisfied(diagnostics: tuple[ConstraintDiagnostic, ...]) -> bool:
    """Require every mandatory optimizer constraint to pass its explicit diagnostic."""
    return all(item.satisfied for item in diagnostics)


def _fingerprint(
    request: PortfolioOptimizationRequest,
    config: PortfolioOptimizationConfig,
) -> str:
    """Hash canonical decisions, covariance, views and optimizer parameters."""
    request_payload = request.model_dump(mode="json")
    config_payload = config.model_dump(mode="json")
    config_payload["sector_limits"] = sorted(
        config_payload["sector_limits"], key=lambda item: item["sector"]
    )
    canonical = json.dumps(
        {"request": request_payload, "config": config_payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _failure(
    request: PortfolioOptimizationRequest,
    config: PortfolioOptimizationConfig,
    fingerprint: str,
    exclusions: list[AssetExclusion],
    ridge: float,
    message: str,
    *,
    status: Literal["INFEASIBLE", "FAILED"],
    iterations: int = 0,
    constraints: tuple[ConstraintDiagnostic, ...] = (),
) -> PortfolioOptimizationResult:
    """Return a non-actionable, traceable result whenever optimization cannot be accepted."""
    return PortfolioOptimizationResult(
        status=status,
        method=config.method,
        config_id=config.config_id,
        config_version=config.version,
        asset_symbols=tuple(asset.asset_symbol for asset in request.assets),
        allocations=tuple(
            TargetAllocation(asset_symbol=asset.asset_symbol, target_weight=0.0)
            for asset in request.assets
        ),
        exclusions=tuple(sorted(exclusions, key=lambda item: item.asset_symbol)),
        constraints=constraints,
        objective_value=None,
        expected_return=None,
        volatility=None,
        sharpe_ratio=None,
        turnover=None,
        estimated_transaction_cost_fraction=None,
        concentration_hhi=None,
        covariance_regularization=ridge,
        solver_success=False,
        solver_message=message[:2000],
        iterations=iterations,
        as_of=request.as_of,
        horizon=request.horizon,
        data_snapshot_id=request.data_snapshot_id,
        covariance_data_snapshot_id=request.covariance_data_snapshot_id,
        decision_fingerprints=tuple(
            asset.decision.decision_fingerprint for asset in request.assets
        ),
        optimization_fingerprint=fingerprint,
    )


__all__ = ["optimize_portfolio"]
