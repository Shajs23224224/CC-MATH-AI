"""Vanilla option pricing, Black-Scholes Greeks and implied-volatility tools."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]

from core.contracts import (
    ExerciseStyle,
    ImpliedVolatilityResult,
    OptionGreeks,
    OptionPriceResult,
    OptionType,
    VolatilitySurface,
    VolatilitySurfacePoint,
)

_EPSILON = 1e-12
_MIN_VOLATILITY = 1e-8
_DEFAULT_MAX_VOLATILITY = 5.0
_OptionTreeMethod = Literal["binomial", "trinomial"]


def _validate_option_inputs(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[float, float, float, float, float, float]:
    """Validate finite market inputs for a vanilla option."""
    values = (
        spot,
        strike,
        maturity_years,
        risk_free_rate,
        dividend_yield,
        volatility,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("option inputs must all be finite")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if maturity_years < 0.0:
        raise ValueError("maturity_years must be non-negative")
    if volatility <= 0.0:
        raise ValueError("volatility must be positive")
    return (
        float(spot),
        float(strike),
        float(maturity_years),
        float(risk_free_rate),
        float(dividend_yield),
        float(volatility),
    )


def _validate_option_type(option_type: str) -> OptionType:
    """Validate the supported vanilla payoff type."""
    if option_type == "call":
        return "call"
    if option_type == "put":
        return "put"
    raise ValueError("option_type must be 'call' or 'put'")


def _validate_exercise_style(exercise_style: str) -> ExerciseStyle:
    """Validate the exercise convention."""
    if exercise_style == "european":
        return "european"
    if exercise_style == "american":
        return "american"
    raise ValueError("exercise_style must be 'european' or 'american'")


def _intrinsic_value(
    spot: float | np.ndarray,
    strike: float,
    option_type: OptionType,
) -> float | np.ndarray:
    """Return the immediate exercise value."""
    if isinstance(spot, np.ndarray):
        if option_type == "call":
            return np.maximum(spot - strike, 0.0)
        return np.maximum(strike - spot, 0.0)
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _black_scholes_value(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: OptionType,
) -> float:
    """Closed-form European option value, without assembling output metadata."""
    if maturity_years == 0.0:
        return float(_intrinsic_value(spot, strike, option_type))
    root_t = math.sqrt(maturity_years)
    sigma_root_t = volatility * root_t
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * maturity_years
    ) / sigma_root_t
    d2 = d1 - sigma_root_t
    discounted_spot = spot * math.exp(-dividend_yield * maturity_years)
    discounted_strike = strike * math.exp(-risk_free_rate * maturity_years)
    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def black_scholes_price(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> OptionPriceResult:
    """Price a European vanilla option with continuous rates and dividend yield."""
    s, k, t, rate, dividend, sigma = _validate_option_inputs(
        spot, strike, maturity_years, risk_free_rate, dividend_yield, volatility
    )
    kind = _validate_option_type(option_type)
    price = _black_scholes_value(s, k, t, rate, dividend, sigma, kind)
    # Small negative values can arise from floating-point cancellation at extreme inputs.
    if not math.isfinite(price) or price < -1e-10:
        raise ValueError("Black-Scholes calculation produced an invalid price")
    return OptionPriceResult(
        method="black_scholes",
        option_type=kind,
        exercise_style="european",
        spot=s,
        strike=k,
        maturity_years=t,
        risk_free_rate=rate,
        dividend_yield=dividend,
        volatility=sigma,
        price=max(price, 0.0),
        methodology=(
            "European Black-Scholes closed form with continuously compounded risk-free rate "
            "and continuous dividend yield; volatility is a decimal per square-root year."
        ),
    )


def _validate_tree_steps(steps: int) -> int:
    """Reject invalid lattice resolution."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return steps


def _tree_price_result(
    method: _OptionTreeMethod,
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    steps: int,
    price: float,
) -> OptionPriceResult:
    """Assemble shared tree metadata."""
    if not math.isfinite(price) or price < -1e-10:
        raise ValueError(f"{method} tree produced an invalid price")
    return OptionPriceResult(
        method=method,
        option_type=option_type,
        exercise_style=exercise_style,
        spot=spot,
        strike=strike,
        maturity_years=maturity_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        price=max(float(price), 0.0),
        tree_steps=steps,
        methodology=(
            f"{method.title()} recombining lattice with {steps} time steps, risk-neutral "
            "probabilities and discounted backward induction; American nodes compare "
            "continuation value with immediate exercise value."
        ),
    )


def binomial_tree_price(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    exercise_style: ExerciseStyle = "european",
    steps: int = 200,
) -> OptionPriceResult:
    """Price European or American vanilla options using the CRR binomial tree."""
    s, k, t, rate, dividend, sigma = _validate_option_inputs(
        spot, strike, maturity_years, risk_free_rate, dividend_yield, volatility
    )
    kind = _validate_option_type(option_type)
    exercise = _validate_exercise_style(exercise_style)
    count = _validate_tree_steps(steps)
    if t == 0.0:
        value = float(_intrinsic_value(s, k, kind))
        return _tree_price_result(
            "binomial", s, k, t, rate, dividend, sigma, kind, exercise, count, value
        )

    dt = t / count
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend) * dt)
    denominator = up - down
    if abs(denominator) <= _EPSILON:
        raise ValueError("binomial tree has degenerate up/down factors")
    probability_up = (growth - down) / denominator
    if not 0.0 <= probability_up <= 1.0:
        raise ValueError("binomial risk-neutral probability is outside [0, 1]; increase steps")
    discount = math.exp(-rate * dt)

    down_moves = np.arange(count + 1, dtype=np.float64)
    terminal_spots = s * np.power(up, count - down_moves) * np.power(down, down_moves)
    values = np.asarray(_intrinsic_value(terminal_spots, k, kind), dtype=np.float64)
    for current_step in range(count - 1, -1, -1):
        values = discount * (probability_up * values[:-1] + (1.0 - probability_up) * values[1:])
        if exercise == "american":
            moves = np.arange(current_step + 1, dtype=np.float64)
            node_spots = s * np.power(up, current_step - moves) * np.power(down, moves)
            values = np.maximum(values, _intrinsic_value(node_spots, k, kind))
    return _tree_price_result(
        "binomial", s, k, t, rate, dividend, sigma, kind, exercise, count, float(values[0])
    )


def trinomial_tree_price(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    exercise_style: ExerciseStyle = "european",
    steps: int = 200,
) -> OptionPriceResult:
    """Price European or American vanilla options with a moment-matched trinomial tree."""
    s, k, t, rate, dividend, sigma = _validate_option_inputs(
        spot, strike, maturity_years, risk_free_rate, dividend_yield, volatility
    )
    kind = _validate_option_type(option_type)
    exercise = _validate_exercise_style(exercise_style)
    count = _validate_tree_steps(steps)
    if t == 0.0:
        value = float(_intrinsic_value(s, k, kind))
        return _tree_price_result(
            "trinomial", s, k, t, rate, dividend, sigma, kind, exercise, count, value
        )

    dt = t / count
    up = math.exp(sigma * math.sqrt(2.0 * dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend) * dt)
    second_moment = math.exp((2.0 * (rate - dividend) + sigma * sigma) * dt)
    moment_matrix = np.asarray(
        [[up, 1.0, down], [up * up, 1.0, down * down], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    probability_up, probability_middle, probability_down = np.linalg.solve(
        moment_matrix,
        np.asarray([growth, second_moment, 1.0], dtype=np.float64),
    )
    probabilities = (probability_up, probability_middle, probability_down)
    if any(probability < -1e-12 or probability > 1.0 + 1e-12 for probability in probabilities):
        raise ValueError("trinomial risk-neutral probabilities are outside [0, 1]; increase steps")
    probability_up, probability_middle, probability_down = (
        max(float(probability), 0.0) for probability in probabilities
    )
    total_probability = probability_up + probability_middle + probability_down
    probability_up /= total_probability
    probability_middle /= total_probability
    probability_down /= total_probability
    discount = math.exp(-rate * dt)

    states = np.arange(-count, count + 1, dtype=np.float64)
    terminal_spots = s * np.power(up, states)
    values = np.asarray(_intrinsic_value(terminal_spots, k, kind), dtype=np.float64)
    for current_step in range(count - 1, -1, -1):
        values = discount * (
            probability_up * values[2:]
            + probability_middle * values[1:-1]
            + probability_down * values[:-2]
        )
        if exercise == "american":
            states = np.arange(-current_step, current_step + 1, dtype=np.float64)
            node_spots = s * np.power(up, states)
            values = np.maximum(values, _intrinsic_value(node_spots, k, kind))
    return _tree_price_result(
        "trinomial", s, k, t, rate, dividend, sigma, kind, exercise, count, float(values[0])
    )


def monte_carlo_price(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    simulations: int = 10_000,
    random_seed: int = 0,
    antithetic: bool = True,
) -> OptionPriceResult:
    """Estimate a European option price with seeded risk-neutral terminal simulations."""
    s, k, t, rate, dividend, sigma = _validate_option_inputs(
        spot, strike, maturity_years, risk_free_rate, dividend_yield, volatility
    )
    kind = _validate_option_type(option_type)
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 100:
        raise ValueError("simulations must be an integer greater than or equal to 100")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if not isinstance(antithetic, bool):
        raise ValueError("antithetic must be a boolean")
    if antithetic and simulations % 2 != 0:
        raise ValueError("antithetic Monte Carlo requires an even simulation count")

    generator = np.random.default_rng(random_seed)
    if t == 0.0:
        terminal_spots = np.full(simulations, s, dtype=np.float64)
    else:
        normal_count = simulations // 2 if antithetic else simulations
        normal_draws = generator.standard_normal(size=normal_count)
        if antithetic:
            normal_draws = np.concatenate((normal_draws, -normal_draws))
        terminal_spots = s * np.exp(
            (rate - dividend - 0.5 * sigma * sigma) * t + sigma * math.sqrt(t) * normal_draws
        )

    payoffs = np.asarray(_intrinsic_value(terminal_spots, k, kind), dtype=np.float64)
    discounted_payoffs = math.exp(-rate * t) * payoffs
    price = float(np.mean(discounted_payoffs))
    if antithetic:
        half = simulations // 2
        paired_means = (discounted_payoffs[:half] + discounted_payoffs[half:]) / 2.0
        standard_error = float(np.std(paired_means, ddof=1) / math.sqrt(half))
    else:
        standard_error = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(simulations))

    if not math.isfinite(price) or not math.isfinite(standard_error):
        raise ValueError("Monte Carlo produced non-finite price or standard error")
    return OptionPriceResult(
        method="monte_carlo",
        option_type=kind,
        exercise_style="european",
        spot=s,
        strike=k,
        maturity_years=t,
        risk_free_rate=rate,
        dividend_yield=dividend,
        volatility=sigma,
        price=max(price, 0.0),
        simulations=simulations,
        random_seed=random_seed,
        antithetic=antithetic,
        standard_error=standard_error,
        methodology=(
            "Risk-neutral lognormal terminal-price simulation with continuous rates and "
            "dividend yield; NumPy Generator seed recorded in output. The standard error "
            "uses paired averages when antithetic variates are enabled."
        ),
    )


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> OptionGreeks:
    """Return analytical Black-Scholes price, Delta, Gamma, Vega, Theta and Rho.

    Vega and Rho are derivatives per one absolute unit of volatility/rate (not per 1%).
    Theta is per year. Delta includes the present-value effect of continuous dividends.
    """
    s, k, t, rate, dividend, sigma = _validate_option_inputs(
        spot, strike, maturity_years, risk_free_rate, dividend_yield, volatility
    )
    kind = _validate_option_type(option_type)
    if t <= 0.0:
        raise ValueError("Black-Scholes Greeks are undefined at expiration")
    root_t = math.sqrt(t)
    sigma_root_t = sigma * root_t
    d1 = (math.log(s / k) + (rate - dividend + 0.5 * sigma * sigma) * t) / sigma_root_t
    d2 = d1 - sigma_root_t
    discounted_spot = s * math.exp(-dividend * t)
    discounted_strike = k * math.exp(-rate * t)
    density = float(norm.pdf(d1))
    gamma = math.exp(-dividend * t) * density / (s * sigma_root_t)
    vega = discounted_spot * density * root_t
    common_theta = -discounted_spot * density * sigma / (2.0 * root_t)
    if kind == "call":
        price = _black_scholes_value(s, k, t, rate, dividend, sigma, kind)
        delta = math.exp(-dividend * t) * float(norm.cdf(d1))
        theta = common_theta - rate * discounted_strike * float(norm.cdf(d2))
        theta += dividend * discounted_spot * float(norm.cdf(d1))
        rho = k * t * math.exp(-rate * t) * float(norm.cdf(d2))
    else:
        price = _black_scholes_value(s, k, t, rate, dividend, sigma, kind)
        delta = math.exp(-dividend * t) * (float(norm.cdf(d1)) - 1.0)
        theta = common_theta + rate * discounted_strike * float(norm.cdf(-d2))
        theta -= dividend * discounted_spot * float(norm.cdf(-d1))
        rho = -k * t * math.exp(-rate * t) * float(norm.cdf(-d2))
    return OptionGreeks(
        option_type=kind,
        spot=s,
        strike=k,
        maturity_years=t,
        risk_free_rate=rate,
        dividend_yield=dividend,
        volatility=sigma,
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        methodology=(
            "Closed-form Black-Scholes partial derivatives with continuous dividend yield; "
            "Vega/Rho are per absolute unit and Theta is annualized."
        ),
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    *,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    min_volatility: float = _MIN_VOLATILITY,
    max_volatility: float = _DEFAULT_MAX_VOLATILITY,
) -> ImpliedVolatilityResult:
    """Invert the European Black-Scholes price with a bracketed Brent solver."""
    s, k, t, rate, dividend, _ = _validate_option_inputs(
        spot, strike, maturity_years, risk_free_rate, dividend_yield, 1.0
    )
    kind = _validate_option_type(option_type)
    if t <= 0.0:
        raise ValueError("implied volatility is undefined at expiration")
    if not math.isfinite(market_price) or market_price < 0.0:
        raise ValueError("market_price must be finite and non-negative")
    if (
        not math.isfinite(min_volatility)
        or not math.isfinite(max_volatility)
        or min_volatility <= 0.0
        or max_volatility <= min_volatility
    ):
        raise ValueError("volatility search bounds must be finite, positive and increasing")

    discounted_spot = s * math.exp(-dividend * t)
    discounted_strike = k * math.exp(-rate * t)
    if kind == "call":
        lower_bound = max(discounted_spot - discounted_strike, 0.0)
        upper_bound = discounted_spot
    else:
        lower_bound = max(discounted_strike - discounted_spot, 0.0)
        upper_bound = discounted_strike
    if not lower_bound < market_price < upper_bound:
        raise ValueError("market_price must be strictly inside no-arbitrage price bounds")

    def difference(volatility_value: float) -> float:
        return _black_scholes_value(s, k, t, rate, dividend, volatility_value, kind) - market_price

    lower_difference = difference(min_volatility)
    upper_difference = difference(max_volatility)
    if lower_difference == 0.0:
        root = min_volatility
        iterations = 1
    elif upper_difference == 0.0:
        root = max_volatility
        iterations = 1
    elif lower_difference * upper_difference > 0.0:
        raise ValueError("implied volatility is not bracketed within the configured search bounds")
    else:
        root, solver = brentq(
            difference,
            min_volatility,
            max_volatility,
            xtol=1e-10,
            full_output=True,
            disp=True,
        )
        if not solver.converged:
            raise ValueError("implied-volatility solver did not converge")
        iterations = int(solver.iterations)

    return ImpliedVolatilityResult(
        option_type=kind,
        spot=s,
        strike=k,
        maturity_years=t,
        risk_free_rate=rate,
        dividend_yield=dividend,
        market_price=market_price,
        implied_volatility=float(root),
        lower_price_bound=float(lower_bound),
        upper_price_bound=float(upper_bound),
        iterations=iterations,
        methodology=(
            "Brent root-finding on the European Black-Scholes pricing error within explicit "
            "volatility bounds; market quote must lie strictly inside no-arbitrage price bounds."
        ),
    )


def build_volatility_surface(
    underlying_symbol: str,
    points: Sequence[VolatilitySurfacePoint],
    *,
    surface_id: str,
    source_version: str,
    as_of: str,
    methodology: str = "Observed implied-volatility quotes; no interpolation or extrapolation.",
) -> VolatilitySurface:
    """Create a canonical sparse implied-volatility surface from validated quotes."""
    if len(points) == 0:
        raise ValueError("at least one volatility surface point is required")
    ordered = tuple(sorted(points, key=lambda item: (item.maturity_years, item.strike)))
    return VolatilitySurface(
        surface_id=surface_id,
        underlying_symbol=underlying_symbol,
        source_version=source_version,
        as_of=as_of,
        points=ordered,
        methodology=methodology,
    )
