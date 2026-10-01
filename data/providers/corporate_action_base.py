from __future__ import annotations

from typing import Protocol

import pandas as pd

from core.contracts import CorporateActionRequest


class CorporateActionProvider(Protocol):
    @property
    def name(self) -> str: ...

    def fetch(self, request: CorporateActionRequest) -> pd.DataFrame: ...
