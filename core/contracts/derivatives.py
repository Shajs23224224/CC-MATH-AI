"""Typed contracts for F25 option prices, Greeks and implied-volatility surfaces."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]
OptionPricingMethod = Literal["black_scholes", "binomial", "trinomial", "monte_carlo"]


class OptionPriceResult(BaseModel):
    """Auditable vanilla option price and numerical method metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    method: OptionPricingMethod
    option_type: OptionType
    exercise_style: ExerciseStyle
    spot: float = Field(gt=0.0)
    strike: float = Field(gt=0.0)
    maturity_years: float = Field(ge=0.0)
    risk_free_rate: float
    dividend_yield: float
    volatility: float = Field(gt=0.0)
    price: float = Field(ge=0.0)
    tree_steps: int | None = Field(default=None, ge=1)
    simulations: int | None = Field(default=None, ge=100)
    random_seed: int | None = Field(default=None, ge=0)
    antithetic: bool | None = None
    standard_error: float | None = Field(default=None, ge=0.0)
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_method_metadata(self) -> OptionPriceResult:
        """Keep exercise style and solver metadata consistent with the method."""
        if self.method == "black_scholes":
            if self.exercise_style != "european":
                raise ValueError(
                    "Black-Scholes closed-form pricing supports European exercise only"
                )
            if any(
                value is not None
                for value in (
                    self.tree_steps,
                    self.simulations,
                    self.random_seed,
                    self.antithetic,
                    self.standard_error,
                )
            ):
                raise ValueError("Black-Scholes output cannot include tree or simulation metadata")
        elif self.method in {"binomial", "trinomial"}:
            if self.tree_steps is None:
                raise ValueError("tree pricing requires tree_steps")
            if any(
                value is not None
                for value in (
                    self.simulations,
                    self.random_seed,
                    self.antithetic,
                    self.standard_error,
                )
            ):
                raise ValueError("tree pricing cannot include Monte Carlo metadata")
        else:
            if self.exercise_style != "european":
                raise ValueError("Monte Carlo pricing in F25 supports European exercise only")
            if self.simulations is None or self.random_seed is None or self.antithetic is None:
                raise ValueError(
                    "Monte Carlo output requires simulations, seed and antithetic flag"
                )
            if self.standard_error is None:
                raise ValueError("Monte Carlo output requires standard_error")
            if self.tree_steps is not None:
                raise ValueError("Monte Carlo output cannot include tree_steps")
        return self


class OptionGreeks(BaseModel):
    """Analytical Black-Scholes price sensitivities with explicit units."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    method: Literal["black_scholes_analytic"] = "black_scholes_analytic"
    option_type: OptionType
    spot: float = Field(gt=0.0)
    strike: float = Field(gt=0.0)
    maturity_years: float = Field(gt=0.0)
    risk_free_rate: float
    dividend_yield: float
    volatility: float = Field(gt=0.0)
    price: float = Field(ge=0.0)
    delta: float
    gamma: float = Field(ge=0.0)
    vega: float = Field(ge=0.0)
    theta: float
    rho: float
    methodology: str = Field(min_length=1)


class ImpliedVolatilityResult(BaseModel):
    """Implied volatility solved from a European option market price."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    option_type: OptionType
    spot: float = Field(gt=0.0)
    strike: float = Field(gt=0.0)
    maturity_years: float = Field(gt=0.0)
    risk_free_rate: float
    dividend_yield: float
    market_price: float = Field(ge=0.0)
    implied_volatility: float = Field(gt=0.0)
    lower_price_bound: float = Field(ge=0.0)
    upper_price_bound: float = Field(gt=0.0)
    iterations: int = Field(ge=1)
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_no_arbitrage_bounds(self) -> ImpliedVolatilityResult:
        """Ensure the quote lies strictly inside the bounds needed for finite IV."""
        if self.lower_price_bound >= self.upper_price_bound:
            raise ValueError("option price bounds are invalid")
        if not self.lower_price_bound < self.market_price < self.upper_price_bound:
            raise ValueError("market_price must be strictly inside no-arbitrage price bounds")
        return self


class VolatilitySurfacePoint(BaseModel):
    """One observed maturity/strike implied-volatility quote."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    maturity_years: float = Field(gt=0.0)
    strike: float = Field(gt=0.0)
    implied_volatility: float = Field(gt=0.0)


class VolatilitySurface(BaseModel):
    """Versioned sparse implied-volatility surface; no interpolation is implied."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    surface_id: str = Field(min_length=1)
    underlying_symbol: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    as_of: str = Field(min_length=1)
    points: tuple[VolatilitySurfacePoint, ...] = Field(min_length=1)
    methodology: str = Field(min_length=1)

    @field_validator("underlying_symbol")
    @classmethod
    def normalize_underlying(cls, value: str) -> str:
        """Canonicalize the underlying ticker."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("underlying_symbol must be non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_surface_points(self) -> VolatilitySurface:
        """Require unique maturity/strike coordinates in deterministic order."""
        coordinates = [(point.maturity_years, point.strike) for point in self.points]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("volatility surface cannot contain duplicate maturity/strike quotes")
        if coordinates != sorted(coordinates):
            raise ValueError("volatility surface points must be sorted by maturity then strike")
        if not all(math.isfinite(point.implied_volatility) for point in self.points):
            raise ValueError("surface volatilities must be finite")
        return self
