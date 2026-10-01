from core.config import AppEnvironment, CMathSettings


def test_default_configuration_is_safe() -> None:
    settings = CMathSettings()
    assert settings.environment == AppEnvironment.DEVELOPMENT
    assert settings.broker.provider == "none"
    assert settings.broker.live_execution_enabled is False


def test_paper_configuration_remains_non_live() -> None:
    settings = CMathSettings(environment=AppEnvironment.PAPER)
    assert settings.broker.live_execution_enabled is False
