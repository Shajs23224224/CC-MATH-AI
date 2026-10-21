"""Deterministic portfolio exposure, liquidity and turnover metrics."""

from __future__ import annotations

import math

from core.contracts import (
    LiquidityRiskMetrics,
    Portfolio,
    PortfolioExposureMetrics,
    RiskMetric,
    TurnoverMetrics,
)


def portfolio_exposure_metrics(portfolio: Portfolio) -> PortfolioExposureMetrics:
    """Calculate gross/net exposure, leverage and absolute-weight concentration.

    Equity is cash plus signed position market values. Gross exposure uses the
    sum of absolute market values. If equity is non-positive, normalized risk
    ratios are undefined and the calculation fails explicitly.
    """
    position_values = tuple(position.market_value for position in portfolio.positions)
    if not all(math.isfinite(value) for value in position_values):
        raise ValueError("position market values must be finite")
    if not math.isfinite(portfolio.cash):
        raise ValueError("portfolio cash must be finite")

    gross_exposure = math.fsum(abs(value) for value in position_values)
    net_exposure = math.fsum(position_values)
    equity = portfolio.cash + net_exposure
    if not math.isfinite(equity) or equity <= 0.0:
        raise ValueError("portfolio equity must be finite and positive")

    weights = (
        tuple(abs(value) / gross_exposure for value in position_values)
        if gross_exposure > 0.0
        else ()
    )
    concentration = math.fsum(weight * weight for weight in weights)
    largest_weight = max(weights, default=0.0)

    return PortfolioExposureMetrics(
        portfolio_id=portfolio.portfolio_id,
        timestamp=portfolio.timestamp,
        base_currency=portfolio.base_currency.upper(),
        equity=equity,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        gross_leverage=gross_exposure / equity,
        net_exposure_ratio=net_exposure / equity,
        cash_weight=portfolio.cash / equity,
        concentration_hhi=concentration,
        largest_position_weight=largest_weight,
        position_count=len(position_values),
    )


def liquidity_risk_metrics(
    asset_symbol: str,
    position_notional: float,
    average_daily_dollar_volume: float,
    max_participation_rate: float = 0.10,
) -> LiquidityRiskMetrics:
    """Estimate liquidation days at a capped share of average daily dollar volume.

    Estimated days equal absolute position notional divided by the allowed daily
    traded notional (average daily dollar volume times participation limit).
    This is an estimate, not an execution guarantee.
    """
    if not asset_symbol.strip():
        raise ValueError("asset_symbol must not be empty")
    if not math.isfinite(position_notional) or position_notional < 0.0:
        raise ValueError("position_notional must be finite and non-negative")
    if (
        not math.isfinite(average_daily_dollar_volume)
        or average_daily_dollar_volume <= 0.0
    ):
        raise ValueError("average_daily_dollar_volume must be finite and positive")
    if (
        not math.isfinite(max_participation_rate)
        or not 0.0 < max_participation_rate <= 1.0
    ):
        raise ValueError("max_participation_rate must be in (0, 1]")

    return LiquidityRiskMetrics(
        asset_symbol=asset_symbol.strip().upper(),
        position_notional=position_notional,
        average_daily_dollar_volume=average_daily_dollar_volume,
        max_participation_rate=max_participation_rate,
        participation_rate=position_notional / average_daily_dollar_volume,
        estimated_liquidation_days=(
            position_notional / (average_daily_dollar_volume * max_participation_rate)
        ),
    )


def turnover_risk_metrics(
    traded_notional: float,
    average_portfolio_value: float,
) -> TurnoverMetrics:
    """Calculate gross traded notional divided by average portfolio value."""
    if not math.isfinite(traded_notional) or traded_notional < 0.0:
        raise ValueError("traded_notional must be finite and non-negative")
    if not math.isfinite(average_portfolio_value) or average_portfolio_value <= 0.0:
        raise ValueError("average_portfolio_value must be finite and positive")
    return TurnoverMetrics(
        traded_notional=traded_notional,
        average_portfolio_value=average_portfolio_value,
        turnover_ratio=traded_notional / average_portfolio_value,
    )


def portfolio_risk_metrics(portfolio: Portfolio) -> tuple[RiskMetric, ...]:
    """Project portfolio exposure metrics into canonical constraint metric names."""
    exposure = portfolio_exposure_metrics(portfolio)
    timestamp = exposure.timestamp.isoformat()
    return (
        RiskMetric(
            name="equity",
            value=exposure.equity,
            unit=exposure.base_currency,
            timestamp=timestamp,
            methodology="cash plus signed position market values",
        ),
        RiskMetric(
            name="gross_exposure",
            value=exposure.gross_exposure,
            unit=exposure.base_currency,
            timestamp=timestamp,
            methodology="sum of absolute position market values",
        ),
        RiskMetric(
            name="net_exposure",
            value=exposure.net_exposure,
            unit=exposure.base_currency,
            timestamp=timestamp,
            methodology="sum of signed position market values",
        ),
        RiskMetric(
            name="gross_leverage",
            value=exposure.gross_leverage,
            unit="multiple_of_equity",
            timestamp=timestamp,
            methodology="gross exposure divided by positive equity",
        ),
        RiskMetric(
            name="net_exposure_ratio",
            value=exposure.net_exposure_ratio,
            unit="multiple_of_equity",
            timestamp=timestamp,
            methodology="net exposure divided by positive equity",
        ),
        RiskMetric(
            name="cash_weight",
            value=exposure.cash_weight,
            unit="fraction_of_equity",
            timestamp=timestamp,
            methodology="cash divided by positive equity",
        ),
        RiskMetric(
            name="concentration_hhi",
            value=exposure.concentration_hhi,
            unit="index_0_to_1",
            timestamp=timestamp,
            methodology="sum of squared absolute-exposure weights",
        ),
        RiskMetric(
            name="largest_position_weight",
            value=exposure.largest_position_weight,
            unit="fraction_of_gross_exposure",
            timestamp=timestamp,
            methodology="largest absolute position divided by gross exposure",
        ),
    )
