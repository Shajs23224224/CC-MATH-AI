from __future__ import annotations

from typing import Protocol

import pandas as pd

from core.contracts import MacroDataRequest


class MacroDataProvider(Protocol):
    """Port implemented by macro-data adapters."""

    @property
    def name(self) -> str: ...

    def fetch(self, request: MacroDataRequest) -> pd.DataFrame: ...
