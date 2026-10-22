# F25 — Derivatives

## Purpose

F25 introduces pricing and sensitivity measurements for vanilla European and American options. It computes advisory model outputs only; no exchange, broker or order-execution integration is added.

## Inputs and units

- `spot` and `strike` are strictly positive and expressed in the same currency/unit.
- `maturity_years` is non-negative time to expiry in years.
- `risk_free_rate` and `dividend_yield` are annual continuously compounded decimals.
- `volatility` is annualized standard deviation in decimal units per square-root year (0.20 means 20%).
- An input return series or currency value is never inferred implicitly. Rates, dividend yield, volatility, model method, time horizon and numerical settings are retained with results.

## Public API

Import from `quant.derivatives`:

- `black_scholes_price(..., option_type="call", dividend_yield=0.0)`
- `binomial_tree_price(..., exercise_style="european", steps=200)`
- `trinomial_tree_price(..., exercise_style="european", steps=200)`
- `monte_carlo_price(..., simulations=10_000, random_seed=0, antithetic=True)`
- `black_scholes_greeks(...)`
- `implied_volatility(market_price, ...)`
- `build_volatility_surface(underlying_symbol, points, surface_id, source_version, as_of)`

## Pricing methods

- **Black–Scholes:** closed form for European vanilla calls and puts with a continuous dividend yield. At expiry the value is intrinsic; Greeks are intentionally undefined at expiry.
- **CRR binomial:** recombining up/down stock lattice with risk-neutral probabilities. European and American exercise are supported; each American node chooses the greater of continuation and immediate exercise value.
- **Moment-matched trinomial:** recombining up/middle/down lattice matches first and second risk-neutral stock-price moments. European and American exercise are supported. If the requested time step yields invalid risk-neutral probabilities, the function raises an error instead of clipping materially invalid probabilities.
- **Monte Carlo:** estimates European option value from risk-neutral lognormal terminal prices. The seed, number of simulations, antithetic-variates setting and estimated standard error are returned. Antithetic simulation requires an even number of paths. American early exercise is not approximated by this terminal-only estimator.

The tree results are numerical approximations; increasing step count typically improves convergence, subject to lattice alignment and probability constraints. Monte Carlo output is an estimate with sampling error, not an exact value.

## Greeks and implied volatility

The analytical Black–Scholes output includes Delta, Gamma, Vega, Theta and Rho.

- **Delta:** price sensitivity to one currency unit of spot.
- **Gamma:** second derivative with respect to spot.
- **Vega:** derivative with respect to volatility per absolute unit (not per 1 percentage point).
- **Theta:** derivative per year of calendar time.
- **Rho:** derivative per absolute unit of the risk-free rate (not per 1 percentage point).

Implied volatility uses a bracketed Brent solver to invert the European Black–Scholes price. Market price must lie strictly inside the no-arbitrage lower and upper bounds, and the root must be bracketed inside the configured volatility search interval. A quote at the intrinsic lower bound may imply zero volatility; a quote at the upper bound may require unbounded volatility. F25 explicitly rejects these edge cases rather than inventing a finite IV.

## Volatility surface

`build_volatility_surface` stores a versioned sparse set of (maturity, strike, implied volatility) quotes and sorts them deterministically. Duplicate maturity/strike coordinates are rejected. It does **not** interpolate, extrapolate, smooth, calibrate a parametric surface or independently prove cross-strike/calendar no-arbitrage; consumers must not treat this quote container as a calibrated risk-neutral surface.

## Limitations and controls

- Vanilla model assumptions omit stochastic volatility, jumps, discrete dividends, early-exercise effects in Black–Scholes, transaction costs and liquidity.
- Risk-neutral pricing depends on the chosen rates, dividends and volatility inputs.
- American exercise is supported by the lattice methods only.
- Greeks depend on model assumptions and may be unstable near expiry or around discontinuous payoffs.
- Implied volatility summarizes a quote under a specific model; it is not itself a forecast of realized volatility.
- No output guarantees a realizable execution price or investment outcome.

## Validation

The dedicated `.github/workflows/f25-validation.yml` workflow runs Ruff lint/format, mypy, numerical reference tests and F21–F24 regression tests. General CI is also run on Python 3.11 and 3.12 with the Security dependency audit and static scan.
