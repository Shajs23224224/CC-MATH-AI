from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketDataQualitySummary:
    """Minimal F07 quality summary; F12 will extend this subsystem."""

    rows: int
    valid_rows: int
    rejected_rows: int

    @property
    def status(self) -> str:
        return "VALID" if self.rejected_rows == 0 else "INVALID"
