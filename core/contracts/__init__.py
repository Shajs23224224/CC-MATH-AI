"""Stable domain contracts used across C-MATH-AI."""

from .backtest import BacktestResult
from .decisions import DecisionRecord, Evidence, Forecast, Signal
from .enums import AssetType, DataQualityStatus, OrderSide, OrderType, RiskAction, SignalType
from .execution import ExecutionReport, Fill, Order
from .fundamentals import FundamentalSnapshot
from .market import Asset, MarketBar, PriceSeries
from .portfolio import Portfolio, Position, SizingResult, TargetAllocation
from .quant import AlgorithmMetadata, AlgorithmResult
from .risk import RiskConstraint, RiskEvaluation, RiskMetric

__all__ = [
    "AlgorithmMetadata",
    "AlgorithmResult",
    "Asset",
    "AssetType",
    "BacktestResult",
    "DataQualityStatus",
    "DecisionRecord",
    "Evidence",
    "ExecutionReport",
    "Fill",
    "Forecast",
    "FundamentalSnapshot",
    "MarketBar",
    "Order",
    "OrderSide",
    "OrderType",
    "Portfolio",
    "Position",
    "PriceSeries",
    "RiskAction",
    "RiskConstraint",
    "RiskEvaluation",
    "RiskMetric",
    "Signal",
    "SignalType",
    "SizingResult",
    "TargetAllocation",
]
