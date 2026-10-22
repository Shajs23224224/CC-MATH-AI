"""Stable domain contracts used across C-MATH-AI."""

from .backtest import BacktestResult
from .cointegration import (
    CointegratingPairResult,
    CointegrationTestResult,
    JohansenTestResult,
    MultipleTestingResult,
    UnitRootTestResult,
)
from .corporate_actions import CorporateAction, CorporateActionBatch, CorporateActionRequest
from .decisions import DecisionRecord, Evidence, Forecast, Signal
from .dependence import DependenceMatrix, PCAResult
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
from .fundamentals import FundamentalGrowthPoint, FundamentalSnapshot
from .macro_data import MacroDataBatch, MacroDataRequest, MacroFrequency, MacroObservation
from .market import Asset, MarketBar, PriceSeries
from .market_data import MarketDataBatch, MarketDataRequest
from .math import OptimizationResult
from .normalization import NormalizationReport, NormalizationStatus
from .portfolio import Portfolio, Position, SizingResult, TargetAllocation
from .quality import (
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
    FeatureDefinition,
    FeatureRecord,
    FeatureSet,
)
from .quant import AlgorithmMetadata, AlgorithmResult
from .returns import PerformanceMetrics, ReturnSeries
from .risk import (
    LiquidityRiskMetrics,
    PortfolioExposureMetrics,
    RiskConstraint,
    RiskEvaluation,
    RiskMetric,
    TurnoverMetrics,
)
from .statistics import StatisticalSummary
from .tail_risk import StressScenario, StressTestResult, TailRiskEstimate
from .technical import TechnicalIndicatorSeries
from .time_series import (
    ResidualDiagnostics,
    TemporalSplit,
    TimeSeriesFamily,
    TimeSeriesFitSummary,
    TimeSeriesForecast,
)
from .valuation import SensitivityMatrix, ValuationAssumptions, ValuationResult
from .volatility import (
    DynamicCorrelationForecast,
    MeanReversionFit,
    MeanReversionForecast,
    VolatilityFamily,
    VolatilityFitSummary,
    VolatilityForecast,
)

__all__ = [
    "AlgorithmMetadata",
    "AlgorithmResult",
    "Asset",
    "AssetType",
    "BacktestResult",
    "CorporateAction",
    "CorporateActionBatch",
    "CorporateActionRequest",
    "CointegratingPairResult",
    "CointegrationTestResult",
    "CorporateActionType",
    "DataQualityIssue",
    "DataQualityReport",
    "DataQualitySeverity",
    "DataQualityStatus",
    "DependenceMatrix",
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
    "FundamentalGrowthPoint",
    "FundamentalSnapshot",
    "JohansenTestResult",
    "LiquidityRiskMetrics",
    "MacroDataBatch",
    "MacroDataRequest",
    "MacroFrequency",
    "MacroObservation",
    "MarketBar",
    "MarketDataBatch",
    "MarketDataRequest",
    "MultipleTestingResult",
    "NormalizationReport",
    "NormalizationStatus",
    "OptimizationResult",
    "Order",
    "OrderSide",
    "OrderType",
    "PCAResult",
    "PerformanceMetrics",
    "Portfolio",
    "PortfolioExposureMetrics",
    "Position",
    "PriceSeries",
    "ReturnSeries",
    "RiskAction",
    "RiskConstraint",
    "RiskEvaluation",
    "RiskMetric",
    "ResidualDiagnostics",
    "Signal",
    "StatisticalSummary",
    "StressScenario",
    "StressTestResult",
    "TailRiskEstimate",
    "TechnicalIndicatorSeries",
    "TemporalSplit",
    "TimeSeriesFamily",
    "TimeSeriesFitSummary",
    "TimeSeriesForecast",
    "SensitivityMatrix",
    "ValuationAssumptions",
    "ValuationResult",
    "DynamicCorrelationForecast",
    "MeanReversionFit",
    "MeanReversionForecast",
    "VolatilityFamily",
    "VolatilityFitSummary",
    "VolatilityForecast",
    "SignalType",
    "SizingResult",
    "TargetAllocation",
    "TurnoverMetrics",
    "UnitRootTestResult",
]
