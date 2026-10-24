"""Versioned policy for the deterministic F42 paper execution simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class PaperExecutionPolicy:
    """Explicit simulation assumptions; never configures a live execution venue."""

    policy_id: str = "paper-default"
    version: str = "1.0.0"
    base_currency: str = "USD"
    commission_bps: float = 1.0
    market_slippage_bps: float = 2.0
    max_quote_age_seconds: int = 30
    max_risk_decision_age_seconds: int = 300
    allow_market_orders: bool = True
    allow_limit_orders: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        if not self.version.strip():
            raise ValueError("version must not be empty")
        currency = self.base_currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("base_currency must be a three-letter code")
        object.__setattr__(self, "base_currency", currency)
        if not isfinite(self.commission_bps) or self.commission_bps < 0:
            raise ValueError("commission_bps must be finite and non-negative")
        if not isfinite(self.market_slippage_bps) or not 0 <= self.market_slippage_bps < 10_000:
            raise ValueError("market_slippage_bps must be finite and in [0, 10000)")
        if (
            not isinstance(self.max_quote_age_seconds, int)
            or isinstance(self.max_quote_age_seconds, bool)
            or self.max_quote_age_seconds < 1
        ):
            raise ValueError("max_quote_age_seconds must be a positive integer")
        if (
            not isinstance(self.max_risk_decision_age_seconds, int)
            or isinstance(self.max_risk_decision_age_seconds, bool)
            or self.max_risk_decision_age_seconds < 1
        ):
            raise ValueError("max_risk_decision_age_seconds must be a positive integer")
