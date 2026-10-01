from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from .settings import CMathSettings


def load_toml_defaults(path: str | Path) -> dict[str, Any]:
    """Load non-secret configuration defaults from a TOML profile.

    Secrets must never be stored in TOML profiles. They are injected through
    environment variables or a dedicated secret manager in deployment.
    """
    profile = Path(path)
    with profile.open("rb") as handle:
        return tomllib.load(handle)


def load_settings_from_environment() -> CMathSettings:
    """Build validated runtime settings from environment variables."""
    return CMathSettings.from_environment()
