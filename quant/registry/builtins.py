"""Curated metadata for existing public quantitative APIs.

This manifest describes known implementations only; it does not claim that the
roadmap's eventual 300+ algorithm catalog is complete.
"""

from __future__ import annotations

from .contracts import (
    AlgorithmLifecycle,
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmPort,
    AlgorithmTestStatus,
)

_PRICE = AlgorithmPort(
    name="prices", dtype="series[float]", description="Chronological price series", unit="price"
)
_RETURNS = AlgorithmPort(
    name="returns", dtype="series[float]", description="Per-period asset returns", unit="ratio"
)
_SCALAR = AlgorithmPort(name="value", dtype="float", description="Computed numeric result")


def _entry(
    algorithm_id: str,
    name: str,
    category: str,
    summary: str,
    method: str,
    inputs: tuple[AlgorithmPort, ...],
    output: AlgorithmPort,
    *,
    implementation_ref: str,
    parameters: tuple[AlgorithmParameter, ...] = (),
    units: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (
        "Input data quality and temporal alignment remain caller responsibilities.",
    ),
) -> AlgorithmMetadata:
    return AlgorithmMetadata(
        algorithm_id=algorithm_id,
        name=name,
        category=category,
        version="1.0.0",
        summary=summary,
        method=method,
        inputs=inputs,
        outputs=(output,),
        parameters=parameters,
        units=units,
        frequencies=("caller-defined",),
        data_requirements=("finite numeric observations",),
        limitations=limitations,
        tags=tags,
        implementation_ref=implementation_ref,
        lifecycle=AlgorithmLifecycle.IMPLEMENTED,
        test_status=AlgorithmTestStatus.NOT_RUN,
    )


_WINDOW = AlgorithmParameter(
    name="window", dtype="int", description="Trailing window size", minimum=1, maximum=5000
)
BUILTIN_ALGORITHMS: tuple[AlgorithmMetadata, ...] = (
    _entry(
        "returns.simple",
        "Simple return",
        "returns",
        "Compute a simple period return from two prices.",
        "P_t / P_(t-1) - 1",
        (_PRICE,),
        _RETURNS,
        implementation_ref="quant.returns.simple_return",
        units=("ratio",),
        tags=("return", "price", "deterministic"),
    ),
    _entry(
        "returns.log",
        "Log return",
        "returns",
        "Compute the natural logarithm of a price ratio.",
        "ln(P_t / P_(t-1))",
        (_PRICE,),
        _RETURNS,
        implementation_ref="quant.returns.log_return",
        units=("log-ratio",),
        tags=("return", "price", "deterministic"),
        limitations=("Prices must be positive.", "Input data must be chronologically aligned."),
    ),
    _entry(
        "returns.cumulative",
        "Cumulative return",
        "returns",
        "Compound a sequence of simple returns.",
        "product(1 + r_t) - 1",
        (_RETURNS,),
        _SCALAR,
        implementation_ref="quant.returns.cumulative_return",
        units=("ratio",),
        tags=("return", "compounding"),
    ),
    _entry(
        "returns.annualized_volatility",
        "Annualized volatility",
        "risk",
        "Annualize return dispersion using a declared periods-per-year convention.",
        "std(returns) * sqrt(periods_per_year)",
        (_RETURNS,),
        _SCALAR,
        implementation_ref="quant.returns.annualized_volatility",
        parameters=(
            AlgorithmParameter(
                name="periods_per_year",
                dtype="float",
                description="Observations per year for this data frequency",
                minimum=0.000001,
            ),
        ),
        units=("annualized return ratio",),
        tags=("risk", "volatility"),
    ),
    _entry(
        "statistics.mean",
        "Arithmetic mean",
        "statistics",
        "Compute the arithmetic mean.",
        "sum(x_i) / n",
        (AlgorithmPort(name="values", dtype="array[float]", description="Numeric observations"),),
        _SCALAR,
        implementation_ref="quant.math.arithmetic_mean",
        tags=("statistics", "descriptive"),
    ),
    _entry(
        "statistics.rolling_mean",
        "Rolling mean",
        "statistics",
        "Compute a trailing rolling mean without future observations.",
        "mean(x[t-window+1:t+1])",
        (
            AlgorithmPort(
                name="values", dtype="series[float]", description="Chronological numeric series"
            ),
        ),
        AlgorithmPort(name="values", dtype="series[float]", description="Trailing mean series"),
        implementation_ref="quant.statistics.rolling_mean",
        parameters=(_WINDOW,),
        tags=("statistics", "rolling", "no-lookahead"),
    ),
    _entry(
        "statistics.rolling_std",
        "Rolling standard deviation",
        "statistics",
        "Compute a trailing rolling standard deviation.",
        "std(x[t-window+1:t+1])",
        (
            AlgorithmPort(
                name="values", dtype="series[float]", description="Chronological numeric series"
            ),
        ),
        AlgorithmPort(
            name="values", dtype="series[float]", description="Trailing standard deviation series"
        ),
        implementation_ref="quant.statistics.rolling_std",
        parameters=(_WINDOW,),
        tags=("statistics", "rolling", "risk", "no-lookahead"),
    ),
    _entry(
        "statistics.rolling_zscore",
        "Rolling z-score",
        "statistics",
        "Standardize observations using a trailing window.",
        "(x_t - rolling_mean) / rolling_std",
        (
            AlgorithmPort(
                name="values", dtype="series[float]", description="Chronological numeric series"
            ),
        ),
        AlgorithmPort(name="values", dtype="series[float]", description="Trailing z-score series"),
        implementation_ref="quant.statistics.rolling_zscore",
        parameters=(
            AlgorithmParameter(
                name="window", dtype="int", description="Window size", minimum=2, maximum=5000
            ),
        ),
        units=("standard deviations",),
        tags=("statistics", "normalization", "no-lookahead"),
    ),
    _entry(
        "timeseries.fit_arima",
        "ARIMA model fitting",
        "time_series",
        "Fit an autoregressive integrated moving-average model to a chronological series.",
        "ARIMA(p,d,q) maximum-likelihood estimation",
        (
            AlgorithmPort(
                name="values", dtype="series[float]", description="Chronological observations"
            ),
        ),
        AlgorithmPort(name="model", dtype="fitted_model", description="Fitted ARIMA model"),
        implementation_ref="quant.time_series.fit_arima",
        parameters=(
            AlgorithmParameter(
                name="p", dtype="int", description="AR order", minimum=0, maximum=20
            ),
            AlgorithmParameter(
                name="d", dtype="int", description="Differencing order", minimum=0, maximum=5
            ),
            AlgorithmParameter(
                name="q", dtype="int", description="MA order", minimum=0, maximum=20
            ),
        ),
        tags=("forecast", "time-series", "statistical"),
        limitations=(
            "A model fit is not evidence of out-of-sample predictive value.",
            "Chronological validation is required.",
        ),
    ),
    _entry(
        "math.present_value",
        "Present value",
        "valuation",
        "Discount a future cash flow.",
        "FV / (1 + r)^n",
        (
            AlgorithmPort(
                name="future_value", dtype="float", description="Future cash flow", unit="currency"
            ),
        ),
        AlgorithmPort(
            name="present_value", dtype="float", description="Discounted cash flow", unit="currency"
        ),
        implementation_ref="quant.math.present_value",
        parameters=(
            AlgorithmParameter(
                name="rate",
                dtype="float",
                description="Per-period discount rate",
                minimum=-0.999999,
            ),
            AlgorithmParameter(
                name="periods",
                dtype="int",
                description="Discount periods",
                minimum=0,
                maximum=100000,
            ),
        ),
        units=("currency",),
        tags=("valuation", "discounting"),
    ),
    _entry(
        "math.normal_cdf",
        "Normal cumulative distribution",
        "probability",
        "Evaluate the standard normal cumulative distribution function.",
        "Phi(x)",
        (AlgorithmPort(name="x", dtype="float", description="Standardized scalar input"),),
        _SCALAR,
        implementation_ref="quant.math.normal_cdf",
        tags=("probability", "distribution"),
    ),
    _entry(
        "technical.rsi",
        "Relative Strength Index",
        "technical",
        "Measure momentum using a bounded relative-strength oscillator.",
        "100 - 100 / (1 + average_gain / average_loss)",
        (
            AlgorithmPort(
                name="close",
                dtype="series[float]",
                description="Chronological close prices",
                unit="price",
            ),
        ),
        AlgorithmPort(
            name="rsi", dtype="series[float]", description="RSI oscillator", unit="index 0-100"
        ),
        implementation_ref="quant.technical.rsi",
        parameters=(
            AlgorithmParameter(
                name="window",
                dtype="int",
                description="Lookback period",
                default=14,
                minimum=2,
                maximum=500,
            ),
        ),
        units=("index 0-100",),
        tags=("technical-analysis", "momentum", "ohlcv"),
        limitations=(
            "Warm-up values and flat-series behavior follow the implementation contract.",
        ),
    ),
)


def builtin_algorithms() -> tuple[AlgorithmMetadata, ...]:
    """Return seed metadata without importing implementation modules."""
    return BUILTIN_ALGORITHMS


__all__ = ["BUILTIN_ALGORITHMS", "builtin_algorithms"]
