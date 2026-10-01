import os

import pytest
from pydantic import SecretStr, ValidationError

from core.config import AppEnvironment, BrokerSettings, CMathSettings


def test_development_defaults_disable_live_execution() -> None:
    settings = CMathSettings()
    assert settings.environment == AppEnvironment.DEVELOPMENT
    assert settings.broker.live_execution_enabled is False


def test_live_execution_is_rejected_outside_production() -> None:
    with pytest.raises(ValidationError):
        CMathSettings(
            environment=AppEnvironment.PAPER,
            broker=BrokerSettings(
                provider="paper",
                live_execution_enabled=True,
            ),
        )


def test_production_live_execution_requires_credentials() -> None:
    with pytest.raises(ValidationError):
        CMathSettings(
            environment=AppEnvironment.PRODUCTION,
            broker=BrokerSettings(
                provider="broker-x",
                live_execution_enabled=True,
            ),
        )


def test_production_live_execution_accepts_credentials() -> None:
    settings = CMathSettings(
        environment=AppEnvironment.PRODUCTION,
        broker=BrokerSettings(
            provider="broker-x",
            api_key=SecretStr("key"),
            api_secret=SecretStr("secret"),
            live_execution_enabled=True,
        ),
    )
    assert settings.broker.api_key is not None


def test_environment_loader_reads_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "provider-x")
    monkeypatch.setenv("LIVE_EXECUTION_ENABLED", "false")

    settings = CMathSettings.from_environment()

    assert settings.environment == AppEnvironment.TESTING
    assert settings.data.provider == "provider-x"
    assert settings.broker.live_execution_enabled is False


def test_invalid_boolean_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_EXECUTION_ENABLED", "maybe")
    with pytest.raises(ValueError):
        CMathSettings.from_environment()


def test_secrets_are_not_exposed_by_secretstr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKET_DATA_API_KEY", "super-secret")
    settings = CMathSettings.from_environment()
    assert str(settings.data.api_key) == "**********"
