"""Contracts for deterministic mathematical computations."""

from pydantic import BaseModel, ConfigDict, Field


class OptimizationResult(BaseModel):
    """Auditable result of a deterministic scalar optimization."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(min_length=1)
    optimum: float
    objective_value: float
    lower_bound: float
    upper_bound: float
    iterations: int = Field(ge=0)
    converged: bool
