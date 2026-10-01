from __future__ import annotations

from typing import Protocol

import pandas as pd

from core.contracts import FundamentalDataRequest


class FundamentalDataProvider(Protocol):
    """Port implemented by fundamental-data adapters."""

    @property
    def name(self) -> str: ...

    def fetch(self, request: FundamentalDataRequest) -> pd.DataFrame: ...
