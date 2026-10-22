"""Typed contracts for tail-risk estimates and versioned stress scenarios."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


TailRiskMethod = Literal[
    "historical",
    "parametric_normal",
    "monte_carlo_normal",
    "evt_peaks_over_threshold",
]


class TailRiskEstimate(BaseModel):
    """Auditable Value-at-Risk and Expected Shortfall estimate.

    VaR and CVaR use signed losses as fractions of portfolio notional: positive
    values are losses and negative values indicate a gain at the tail threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    method: TailRiskMethod
    var: float
    cvar: float
    confidence_level: float = Field(gt=0.5, lt=1.0)
    horizon: int = Field(ge=1)
    observations: int = Field(ge=1)
    simulations: int | None = Field(default=None, ge=1)
    random_seed: int | None = Field(default=None, ge=0)
    unit: Literal["fraction_of_notional"] = "fraction_of_notional"
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tail_estimate(self) -> TailRiskEstimate:
        """Enforce consistency between simulation metadata and the method."""
        if self.cvar + 1e-12 < self.var:
            raise ValueError("cvar must be greater than or equal to var")
        if self.method == "monte_carlo_normal":
            if self.simulations is None or self.random_seed is None:
                raise ValueError("Monte Carlo estimates require simulations and random_seed")
        elif self.simulations is not None or self.random_seed is not None:
            raise ValueError("simulation metadata is only valid for Monte Carlo estimates")
        return self


class StressScenario(BaseModel):
    """Versioned asset-return shocks used for deterministic portfolio stress tests."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    scenario_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    shocks: dict[str, float] = Field(min_length=1)

    @field_validator("shocks")
    @classmethod
    def validate_shocks(cls, shocks: dict[str, float]) -> dict[str, float]:
        """Normalize symbols and reject duplicate or impossible simple-return shocks."""
        normalized: dict[str, float] = {}
        for symbol, shock in shocks.items():
            key = symbol.strip().upper()
            if not key:
                raise ValueError("stress shock symbols must be non-empty")
            if key in normalized:
                raise ValueError(f"duplicate stress shock symbol after normalization: {key}")
            if not math.isfinite(shock) or shock <= -1.0:
                raise ValueError("stress shocks must be finite and greater than -1.0")
            normalized[key] = float(shock)
        return dict(sorted(normalized.items()))


class StressTestResult(BaseModel):
    """Deterministic result for one versioned stress scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    scenario_id: str = Field(min_length=1)
    scenario_version: str = Field(min_length=1)
    portfolio_return: float
    portfolio_loss: float
    covered_positions: int = Field(ge=1)
    unit: Literal["fraction_of_equity"] = "fraction_of_equity"
    methodology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_return_loss_sign(self) -> StressTestResult:
        """Keep reported P&L and loss sign conventions aligned."""
        if not math.isclose(self.portfolio_loss, -self.portfolio_return, abs_tol=1e-12):
            raise ValueError("portfolio_loss must equal -portfolio_return")
        return self
