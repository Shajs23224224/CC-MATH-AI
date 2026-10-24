"""Stable request/response contracts for the F43 application boundary."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backtest.engine import Bar
from backtest.robustness.contracts import MonteCarloConfig
from core.contracts import DependenceMatrix, RiskMetric
from execution.paper import ExecutionQuote, PaperExecutionPolicy, PaperOrderRequest
from portfolio.optimization import (
    PortfolioAssetInput,
    PortfolioOptimizationConfig,
    PortfolioOptimizationRequest,
)
from portfolio.sizing import PositionSizingConfig, PositionSizingRequest, SizingAssetInput
from risk.governor import (
    GovernorAssetContext,
    PairwiseCorrelation,
    RiskGovernorConfig,
    RiskGovernorRequest,
)
from signals.decision import DecisionConfig, DecisionOutput, DecisionRequest

JobState = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]
StageState = Literal["SUCCEEDED", "SKIPPED", "FAILED"]


class MarketBarInput(BaseModel):
    """One point-in-time OHLCV observation for the F39 single-series engine."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    timestamp: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(default=0.0, ge=0.0)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("market-data timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_ohlc(self) -> MarketBarInput:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be at least open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be at most open, close and high")
        return self

    def to_bar(self) -> Bar:
        """Convert into the shared, dependency-light F39 OHLCV contract."""
        return Bar(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


class TargetPositionInput(BaseModel):
    """Caller-supplied strategy output; F43 never fabricates signals from future bars."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    timestamp: datetime
    target_units: int = Field(strict=True)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("target timestamps must be timezone-aware")
        return value


class PlatformBacktestConfig(BaseModel):
    """Validated, serializable settings translated to the F39 dataclass."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    initial_capital: float = Field(default=100_000.0, gt=0.0)
    commission_bps: float = Field(default=1.0, ge=0.0)
    slippage_bps: float = Field(default=2.0, ge=0.0)
    periods_per_year: int = Field(default=252, ge=1)


class PlatformWalkForwardConfig(BaseModel):
    """Optional single-step or multi-window causal OOS split configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    initial_train_size: int | None = Field(default=None, ge=2)
    test_size: int | None = Field(default=None, ge=2)
    mode: Literal["rolling", "expanding", "anchored"] = "expanding"
    gap: int = Field(default=0, ge=0)
    anchor_index: int = Field(default=0, ge=0)
    strategy_version: str = Field(default="last-known-target-baseline-v1", min_length=1)

    @model_validator(mode="after")
    def validate_sizes(self) -> PlatformWalkForwardConfig:
        if (self.initial_train_size is None) != (self.test_size is None):
            raise ValueError("initial_train_size and test_size must be set together")
        return self


class PlatformScenarioInput(BaseModel):
    """Serializable adverse scenario translated to the F41 immutable contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=120)
    volatility_scale: float = Field(default=1.0, gt=0.0, le=5.0)
    price_shock_bps: float = Field(default=0.0, gt=-10_000.0, le=10_000.0)
    shock_start_index: int = Field(default=1, ge=1)
    extra_commission_bps: float = Field(default=0.0, ge=0.0)
    extra_slippage_bps: float = Field(default=0.0, ge=0.0)
    execution_delay_bars: int = Field(default=0, ge=0)


class PortfolioAssetMetadata(BaseModel):
    """Point-in-time price, liquidity and exposure inputs aligned to a portfolio asset."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    asset_symbol: str = Field(min_length=1, max_length=80)
    sector: str = Field(min_length=1, max_length=100)
    current_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    eligible: bool = True
    price: float = Field(gt=0.0)
    volatility: float = Field(ge=0.0, le=10.0)
    average_daily_volume: float = Field(gt=0.0)
    average_daily_dollar_volume: float = Field(gt=0.0)
    lot_size: float = Field(default=1.0, gt=0.0)
    stop_loss_price: float | None = Field(default=None, gt=0.0)
    current_notional: float = Field(default=0.0, ge=0.0)
    factor_loadings: dict[str, float] = Field(default_factory=dict)

    @field_validator("asset_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("asset_symbol must not be blank")
        return normalized

    @field_validator("sector")
    @classmethod
    def normalize_sector(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("sector must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_limits(self) -> PortfolioAssetMetadata:
        if self.minimum_weight > self.maximum_weight:
            raise ValueError("minimum_weight must not exceed maximum_weight")
        if self.stop_loss_price is not None and self.stop_loss_price >= self.price:
            raise ValueError("stop_loss_price must be below price for long-only sizing")
        if any(not math.isfinite(value) for value in self.factor_loadings.values()):
            raise ValueError("factor loadings must be finite")
        return self


class PortfolioPipelineInput(BaseModel):
    """Optional F35→F36→F37→F38 request using the real module contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    decisions: tuple[DecisionRequest, ...] = Field(min_length=2, max_length=500)
    decision_config: DecisionConfig = DecisionConfig()
    assets: tuple[PortfolioAssetMetadata, ...] = Field(min_length=2, max_length=500)
    covariance: DependenceMatrix
    horizon: str = Field(min_length=1, max_length=80)
    covariance_as_of: datetime
    covariance_data_snapshot_id: str = Field(min_length=1, max_length=200)
    optimization_config: PortfolioOptimizationConfig = PortfolioOptimizationConfig()
    sizing_config: PositionSizingConfig = PositionSizingConfig()
    equity: float = Field(gt=0.0)
    base_currency: str = Field(default="USD", min_length=3, max_length=3)
    sizing_as_of: datetime
    market_data_as_of: datetime
    risk_snapshot_as_of: datetime
    risk_snapshot_id: str = Field(min_length=1, max_length=200)
    model_snapshot_id: str = Field(min_length=1, max_length=200)
    risk_metrics: tuple[RiskMetric, ...] = Field(default=(), max_length=1000)
    pairwise_correlations: tuple[PairwiseCorrelation, ...] = Field(default=(), max_length=124750)

    @field_validator(
        "covariance_as_of", "sizing_as_of", "market_data_as_of", "risk_snapshot_as_of"
    )
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("portfolio pipeline timestamps must be timezone-aware")
        return value

    @field_validator("base_currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("base_currency must be a three-letter currency code")
        return normalized

    @model_validator(mode="after")
    def validate_alignment(self) -> PortfolioPipelineInput:
        symbols = tuple(item.asset_symbol for item in self.assets)
        if len(set(symbols)) != len(symbols):
            raise ValueError("portfolio metadata symbols must be unique")
        if symbols != self.covariance.assets:
            raise ValueError("portfolio metadata must match sorted covariance asset order")
        decision_symbols = tuple(item.ensemble.asset_symbol.strip().upper() for item in self.decisions)
        if decision_symbols != symbols:
            raise ValueError("F35 decision inputs must exactly match portfolio asset order")
        if any(item.ensemble.horizon != self.horizon for item in self.decisions):
            raise ValueError("F35 ensemble horizons must match the portfolio horizon")
        if self.covariance_as_of > self.sizing_as_of:
            raise ValueError("covariance_as_of cannot be later than sizing_as_of")
        if self.market_data_as_of > self.sizing_as_of:
            raise ValueError("market_data_as_of cannot be later than sizing_as_of")
        if any(item.ensemble.as_of > self.sizing_as_of for item in self.decisions):
            raise ValueError("F35 decisions cannot be later than sizing_as_of")
        if any(item.ensemble.as_of > self.risk_snapshot_as_of for item in self.decisions):
            raise ValueError("risk snapshot cannot precede the F35 decision cutoff")
        if self.covariance.kind != "covariance":
            raise ValueError("F36 requires a covariance matrix")
        if any(
            not isfinite(item.price)
            or not isfinite(item.volatility)
            or not isfinite(item.average_daily_volume)
            or not isfinite(item.average_daily_dollar_volume)
            for item in self.assets
        ):
            raise ValueError("asset price and liquidity inputs must be finite")
        pair_keys = tuple(item.pair_key for item in self.pairwise_correlations)
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("pairwise correlations must be unique")
        for pair in self.pairwise_correlations:
            if pair.asset_a not in symbols or pair.asset_b not in symbols:
                raise ValueError("pairwise correlation references an unknown asset")
        metric_names = tuple(item.name for item in self.risk_metrics)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("risk metrics must have unique names")
        return self


class PlatformRunRequest(BaseModel):
    """Idempotent, bounded research-to-paper job request.

    Strategy targets are caller-provided outputs from a strategy or F35 decision adapter.
    An optional F38 request is evaluated internally before any paper order can be accepted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    job_id: str = Field(min_length=8, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    dataset_id: str = Field(min_length=1, max_length=200)
    symbol: str = Field(min_length=1, max_length=80)
    as_of: datetime
    bars: tuple[MarketBarInput, ...] = Field(min_length=2, max_length=50_000)
    targets: tuple[TargetPositionInput, ...] = Field(default=(), max_length=50_000)
    backtest: PlatformBacktestConfig = PlatformBacktestConfig()
    monte_carlo: MonteCarloConfig = MonteCarloConfig(simulations=250, seed=42)
    adversarial_scenarios: tuple[PlatformScenarioInput, ...] = Field(
        default=(), max_length=100
    )
    walk_forward: PlatformWalkForwardConfig = PlatformWalkForwardConfig()
    portfolio_pipeline: PortfolioPipelineInput | None = None
    risk_request: RiskGovernorRequest | None = None
    risk_policy: RiskGovernorConfig = RiskGovernorConfig()
    paper_orders: tuple[PaperOrderRequest, ...] = Field(default=(), max_length=1_000)
    paper_quotes: tuple[ExecutionQuote, ...] = Field(default=(), max_length=50_000)
    paper_account_id: str = Field(default="f43-paper-account", min_length=1, max_length=160)
    paper_initial_cash: float | None = Field(default=None, gt=0.0)
    paper_policy: PaperExecutionPolicy = PaperExecutionPolicy()

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be blank")
        return normalized

    @field_validator("as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_pipeline_inputs(self) -> PlatformRunRequest:
        if len(self.bars) > 50_000:
            raise ValueError("bars exceed the platform request resource limit")
        timestamps = tuple(item.timestamp for item in self.bars)
        if any(left >= right for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("market bars must have strictly increasing timestamps")
        if timestamps[-1] > self.as_of:
            raise ValueError("market data cannot contain bars later than as_of")
        bar_timestamps = set(timestamps)
        target_timestamps = tuple(item.timestamp for item in self.targets)
        if len(target_timestamps) != len(set(target_timestamps)):
            raise ValueError("target timestamps must be unique")
        if set(target_timestamps) - bar_timestamps:
            raise ValueError("every strategy target timestamp must match an input bar")
        scenario_names = tuple(item.name for item in self.adversarial_scenarios)
        if len(scenario_names) != len(set(scenario_names)):
            raise ValueError("adversarial scenario names must be unique")
        if self.portfolio_pipeline is not None and self.risk_request is not None:
            raise ValueError("provide either portfolio_pipeline or risk_request, not both")
        if self.paper_orders and self.risk_request is None and self.portfolio_pipeline is None:
            raise ValueError("paper orders require an F38 request or an F35-F38 portfolio pipeline")
        if self.paper_orders and not self.paper_quotes:
            raise ValueError("paper orders require at least one execution quote")
        if self.paper_quotes and not self.paper_orders:
            raise ValueError("paper quotes are only accepted when paper orders are requested")
        if self.risk_request is not None and self.risk_request.as_of > self.as_of:
            raise ValueError("risk request as_of cannot be later than platform as_of")
        if self.portfolio_pipeline is not None:
            pipeline = self.portfolio_pipeline
            if any(
                timestamp > self.as_of
                for timestamp in (
                    pipeline.covariance_as_of,
                    pipeline.sizing_as_of,
                    pipeline.market_data_as_of,
                    pipeline.risk_snapshot_as_of,
                )
            ):
                raise ValueError("portfolio pipeline evidence cannot be later than platform as_of")
            if any(item.ensemble.as_of > self.as_of for item in pipeline.decisions):
                raise ValueError("F35 decisions cannot be later than platform as_of")
            if len(pipeline.assets) != len(pipeline.decisions):
                raise ValueError("F35 decisions and portfolio metadata must have equal lengths")
        if self.walk_forward.enabled and len(self.bars) < 4 and (
            self.walk_forward.initial_train_size is not None
        ):
            raise ValueError("custom walk-forward requires at least four market bars")
        if self.walk_forward.initial_train_size is not None:
            if (
                self.walk_forward.initial_train_size
                + self.walk_forward.gap
                + self.walk_forward.test_size
                > len(self.bars)
            ):
                raise ValueError(
                    "walk-forward train, gap and test sizes exceed market-bar count"
                )
        return self


class PlatformStageResult(BaseModel):
    """State and non-sensitive summary for one orchestration stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str = Field(min_length=1)
    state: StageState
    summary: dict[str, Any] = Field(default_factory=dict)


class PlatformJobRecord(BaseModel):
    """Persisted job state exposed by CLI and HTTP callers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: int = Field(ge=0)
    version: int = Field(ge=1)
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


__all__ = [
    "MarketBarInput",
    "PlatformBacktestConfig",
    "PlatformJobRecord",
    "PlatformRunRequest",
    "PlatformScenarioInput",
    "PlatformStageResult",
    "PlatformWalkForwardConfig",
    "TargetPositionInput",
]
