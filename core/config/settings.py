from __future__ import annotations

import os
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class AppEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PAPER = "paper"
    PRODUCTION = "production"


class DataSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(default="local", min_length=1)
    api_key: SecretStr | None = None
    database_url: SecretStr | None = None


class BrokerSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(default="none", min_length=1)
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None
    live_execution_enabled: bool = False


class RiskSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_position_weight: float = Field(default=0.10, gt=0, le=1)
    max_portfolio_leverage: float = Field(default=1.0, gt=0)
    max_daily_loss: float = Field(default=0.02, gt=0, le=1)


class CMathSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: AppEnvironment = AppEnvironment.DEVELOPMENT
    log_level: str = Field(default="INFO", min_length=1)
    data: DataSettings = DataSettings()
    broker: BrokerSettings = BrokerSettings()
    risk: RiskSettings = RiskSettings()

    @model_validator(mode="after")
    def enforce_environment_safety(self) -> Self:
        if self.environment != AppEnvironment.PRODUCTION and self.broker.live_execution_enabled:
            raise ValueError("live execution is only allowed in production")
        if self.environment == AppEnvironment.PAPER and self.broker.live_execution_enabled:
            raise ValueError("paper environment cannot enable live execution")
        if self.environment == AppEnvironment.PRODUCTION and self.broker.live_execution_enabled:
            if self.broker.provider == "none":
                raise ValueError("production live execution requires a broker provider")
            if self.broker.api_key is None or self.broker.api_secret is None:
                raise ValueError("production live execution requires broker credentials")
        return self

    @classmethod
    def from_environment(cls) -> "CMathSettings":
        env = AppEnvironment(os.getenv("APP_ENV", AppEnvironment.DEVELOPMENT.value))

        data = DataSettings(
            provider=os.getenv("MARKET_DATA_PROVIDER", "local"),
            api_key=_secret_from_env("MARKET_DATA_API_KEY"),
            database_url=_secret_from_env("DATABASE_URL"),
        )
        broker = BrokerSettings(
            provider=os.getenv("BROKER_PROVIDER", "none"),
            api_key=_secret_from_env("BROKER_API_KEY"),
            api_secret=_secret_from_env("BROKER_API_SECRET"),
            live_execution_enabled=_env_bool("LIVE_EXECUTION_ENABLED", False),
        )
        risk = RiskSettings(
            max_position_weight=float(os.getenv("RISK_MAX_POSITION_WEIGHT", "0.10")),
            max_portfolio_leverage=float(os.getenv("RISK_MAX_PORTFOLIO_LEVERAGE", "1.0")),
            max_daily_loss=float(os.getenv("RISK_MAX_DAILY_LOSS", "0.02")),
        )
        return cls(
            environment=env,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            data=data,
            broker=broker,
            risk=risk,
        )


def _secret_from_env(name: str) -> SecretStr | None:
    value = os.getenv(name)
    return SecretStr(value) if value else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
