from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BacktestResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str = Field(min_length=1)
    start_date: date
    end_date: date
    initial_capital: float = Field(gt=0)
    final_equity: float = Field(ge=0)
    total_return: float
    max_drawdown: float = Field(ge=0, le=1)
    sharpe_ratio: float
    sortino_ratio: float
    turnover: float = Field(ge=0)
    trades: int = Field(ge=0)
    transaction_costs: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_period(self) -> "BacktestResult":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self
