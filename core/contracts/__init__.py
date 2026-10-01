"""Stable domain contracts used across C-MATH-AI."""

from .backtest import BacktestResult
from .corporate_actions import CorporateAction, CorporateActionBatch, CorporateActionRequest
from .decisions import DecisionRecord, Evidence, Forecast, Signal
from .enums import (
    AssetType,
    CorporateActionType,
    DataQualityStatus,
    Frequency,
    OrderSide,
    OrderType,
    RiskAction,
    SignalType,
)
from .execution import ExecutionReport, Fill, Order
from .fundamental_data import FundamentalDataBatch, FundamentalDataRequest
from .fundamentals import FundamentalSnapshot
from .macro_data import MacroDataBatch, MacroDataRequest, MacroFrequency, MacroObservation
from .market import Asset, MarketBar, PriceSeries
from .market_data import MarketDataBatch, MarketDataRequest
from .math import OptimizationResult
from .normalization import NormalizationReport, NormalizationStatus
from .portfolio import Portfolio, Position, SizingResult, TargetAllocation
from .quant import AlgorithmMetadata, AlgorithmResult
from .quality import (
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
    FeatureDefinition,
    FeatureRecord,
    FeatureSet,
)
from .returns import PerformanceMetrics, ReturnSeries
from .risk import RiskConstraint, RiskEvaluation, RiskMetric

__all__ = [
    "AlgorithmMetadata",
    "AlgorithmResult",
    "Asset",
    "AssetType",
    "BacktestResult",
    "CorporateAction",
    "CorporateActionBatch",
    "CorporateActionRequest",
    "CorporateActionType",
    "DataQualityIssue",
    "DataQualityReport",
    "DataQualitySeverity",
    "DataQualityStatus",
    "DecisionRecord",
    "Evidence",
    "ExecutionReport",
    "FeatureDefinition",
    "FeatureRecord",
    "FeatureSet",
    "Fill",
    "Forecast",
    "Frequency",
    "FundamentalDataBatch",
    "FundamentalDataRequest",
    "FundamentalSnapshot",
    "MacroDataBatch",
    "MacroDataRequest",
    "MacroFrequency",
    "MacroObservation",
    "MarketBar",
    "MarketDataBatch",
    "MarketDataRequest",
    "NormalizationReport",
    "NormalizationStatus",
    "OptimizationResult",
    "Order",
    "OrderSide",
    "OrderType",
    "PerformanceMetrics",
    "Portfolio",
    "Position",
    "PriceSeries",
    "ReturnSeries",
    "RiskAction",
    "RiskConstraint",
    "RiskEvaluation",
    "RiskMetric",
    "Signal",
    "SignalType",
    "SizingResult",
    "TargetAllocation",
]
