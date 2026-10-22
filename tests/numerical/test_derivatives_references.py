"""Numerical reference tests for F25 derivative pricing and Greeks."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from core.contracts import (
    OptionPriceResult,
    VolatilitySurface,
    VolatilitySurfacePoint,
)
from quant.derivatives import (
    binomial_tree_price,
    black_scholes_greeks,
    black_scholes_price,
    build_volatility_surface,
    implied_volatility,
    monte_carlo_price,
    trinomial_tree_price,
)


def test_black_scholes_call_and_put_reference_values() -> None:
    call = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="call")
    put = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="put")

    assert call.price == pytest.approx(10.450583572185565, abs=1e-10)
    assert put.price == pytest.approx(5.573526022256971, abs=1e-10)
    assert call.exercise_style == "european"
    assert call.method == "black_scholes"


def test_put_call_parity_accounts_for_continuous_dividend_yield() -> None:
    spot, strike, maturity, rate, dividend, volatility = 105.0, 100.0, 0.75, 0.04, 0.015, 0.27
    call = black_scholes_price(
        spot, strike, maturity, rate, volatility, option_type="call", dividend_yield=dividend
    )
    put = black_scholes_price(
        spot, strike, maturity, rate, volatility, option_type="put", dividend_yield=dividend
    )

    expected = spot * math.exp(-dividend * maturity) - strike * math.exp(-rate * maturity)
    assert call.price - put.price == pytest.approx(expected, abs=1e-10)


def test_black_scholes_greeks_match_finite_differences() -> None:
    spot, strike, maturity, rate, dividend, volatility = 103.0, 100.0, 0.9, 0.035, 0.01, 0.24
    greeks = black_scholes_greeks(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        option_type="call",
        dividend_yield=dividend,
    )
    h_spot = 1e-3
    h_vol = 1e-4
    price_up = black_scholes_price(
        spot + h_spot,
        strike,
        maturity,
        rate,
        volatility,
        option_type="call",
        dividend_yield=dividend,
    ).price
    price_down = black_scholes_price(
        spot - h_spot,
        strike,
        maturity,
        rate,
        volatility,
        option_type="call",
        dividend_yield=dividend,
    ).price
    vol_up = black_scholes_price(
        spot,
        strike,
        maturity,
        rate,
        volatility + h_vol,
        option_type="call",
        dividend_yield=dividend,
    ).price
    vol_down = black_scholes_price(
        spot,
        strike,
        maturity,
        rate,
        volatility - h_vol,
        option_type="call",
        dividend_yield=dividend,
    ).price

    assert greeks.delta == pytest.approx((price_up - price_down) / (2 * h_spot), abs=1e-6)
    assert greeks.gamma == pytest.approx(
        (price_up - 2 * greeks.price + price_down) / (h_spot * h_spot), abs=1e-4
    )
    assert greeks.vega == pytest.approx((vol_up - vol_down) / (2 * h_vol), abs=1e-5)
    assert greeks.gamma > 0.0
    assert greeks.vega > 0.0
    assert greeks.rho > 0.0
    put_greeks = black_scholes_greeks(spot, strike, maturity, rate, volatility, option_type="put", dividend_yield=dividend)
    assert put_greeks.delta < 0.0
    assert put_greeks.rho < 0.0


def test_binomial_and_trinomial_trees_converge_toward_black_scholes() -> None:
    reference = black_scholes_price(100.0, 105.0, 1.25, 0.03, 0.22, option_type="call")
    binomial = binomial_tree_price(100.0, 105.0, 1.25, 0.03, 0.22, steps=400)
    trinomial = trinomial_tree_price(100.0, 105.0, 1.25, 0.03, 0.22, steps=200)

    assert abs(binomial.price - reference.price) < 0.05
    assert abs(trinomial.price - reference.price) < 0.05
    assert binomial.tree_steps == 400
    assert trinomial.tree_steps == 200


def test_american_put_tree_dominates_corresponding_european_put() -> None:
    european = binomial_tree_price(
        90.0, 100.0, 1.0, 0.08, 0.25, option_type="put", exercise_style="european", steps=250
    )
    american = binomial_tree_price(
        90.0, 100.0, 1.0, 0.08, 0.25, option_type="put", exercise_style="american", steps=250
    )

    assert american.price >= european.price
    assert american.exercise_style == "american"


def test_monte_carlo_is_seeded_and_agrees_with_analytic_price_within_sampling_error() -> None:
    reference = black_scholes_price(
        100.0, 100.0, 1.0, 0.04, 0.23, option_type="call", dividend_yield=0.01
    )
    first = monte_carlo_price(
        100.0,
        100.0,
        1.0,
        0.04,
        0.23,
        option_type="call",
        dividend_yield=0.01,
        simulations=20_000,
        random_seed=45,
        antithetic=True,
    )
    second = monte_carlo_price(
        100.0,
        100.0,
        1.0,
        0.04,
        0.23,
        option_type="call",
        dividend_yield=0.01,
        simulations=20_000,
        random_seed=45,
        antithetic=True,
    )

    assert first == second
    assert first.standard_error is not None
    assert abs(first.price - reference.price) <= 4.0 * first.standard_error + 1e-8
    assert first.random_seed == 45
    assert first.simulations == 20_000


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_recovers_input_volatility(option_type: str) -> None:
    market = black_scholes_price(
        125.0, 110.0, 0.65, 0.035, 0.31, option_type=option_type, dividend_yield=0.012
    )
    result = implied_volatility(
        market.price,
        125.0,
        110.0,
        0.65,
        0.035,
        option_type=option_type,
        dividend_yield=0.012,
    )

    assert result.implied_volatility == pytest.approx(0.31, abs=1e-7)
    assert result.lower_price_bound < result.market_price < result.upper_price_bound
    assert result.iterations >= 1


def test_implied_volatility_rejects_quotes_outside_no_arbitrage_bounds() -> None:
    with pytest.raises(ValueError, match="strictly inside no-arbitrage"):
        implied_volatility(0.0, 100.0, 100.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="expiration"):
        implied_volatility(1.0, 100.0, 100.0, 0.0, 0.05)


def test_expired_tree_options_equal_intrinsic_value() -> None:
    call = binomial_tree_price(105.0, 100.0, 0.0, 0.03, 0.2, option_type="call")
    put = trinomial_tree_price(95.0, 100.0, 0.0, 0.03, 0.2, option_type="put")
    analytic = black_scholes_price(105.0, 100.0, 0.0, 0.03, 0.2, option_type="call")

    assert call.price == pytest.approx(5.0)
    assert put.price == pytest.approx(5.0)
    assert analytic.price == pytest.approx(5.0)


def test_volatility_surface_normalizes_and_sorts_sparse_quotes() -> None:
    points = (
        VolatilitySurfacePoint(maturity_years=1.0, strike=110.0, implied_volatility=0.28),
        VolatilitySurfacePoint(maturity_years=0.5, strike=100.0, implied_volatility=0.24),
        VolatilitySurfacePoint(maturity_years=0.5, strike=90.0, implied_volatility=0.29),
    )
    surface = build_volatility_surface(
        " aapl ",
        points,
        surface_id="AAPL-IV-2026-10-22",
        source_version="quotes-v1",
        as_of="2026-10-22T16:00:00Z",
    )

    assert isinstance(surface, VolatilitySurface)
    assert surface.underlying_symbol == "AAPL"
    assert [(point.maturity_years, point.strike) for point in surface.points] == [
        (0.5, 90.0),
        (0.5, 100.0),
        (1.0, 110.0),
    ]


def test_surface_rejects_duplicate_maturity_strike_quote() -> None:
    point = VolatilitySurfacePoint(maturity_years=1.0, strike=100.0, implied_volatility=0.2)
    with pytest.raises(ValidationError, match="duplicate maturity/strike"):
        build_volatility_surface(
            "AAPL",
            (point, point),
            surface_id="surface",
            source_version="1",
            as_of="2026-10-22",
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: black_scholes_price(0.0, 100.0, 1.0, 0.05, 0.2), "spot and strike"),
        (lambda: black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0), "volatility must be positive"),
        (lambda: binomial_tree_price(100.0, 100.0, 1.0, 0.05, 0.2, steps=0), "steps must"),
        (
            lambda: monte_carlo_price(100.0, 100.0, 1.0, 0.05, 0.2, simulations=99),
            "simulations must",
        ),
        (
            lambda: monte_carlo_price(100.0, 100.0, 1.0, 0.05, 0.2, simulations=101),
            "even simulation count",
        ),
        (lambda: black_scholes_greeks(100.0, 100.0, 0.0, 0.05, 0.2), "undefined at expiration"),
    ],
)
def test_invalid_derivative_inputs_fail_explicitly(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


def test_option_price_contract_requires_method_metadata() -> None:
    with pytest.raises(ValidationError, match="tree pricing requires tree_steps"):
        OptionPriceResult(
            method="binomial",
            option_type="call",
            exercise_style="european",
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
            price=10.0,
            methodology="invalid contract fixture",
        )
