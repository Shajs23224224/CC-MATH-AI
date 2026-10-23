"""Curated metadata for existing public quantitative APIs.

This manifest describes known implementations only; it does not claim that the
roadmap's eventual 300+ algorithm catalog is complete.
"""

from __future__ import annotations

from typing import cast

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
_SEED_ALGORITHMS: tuple[AlgorithmMetadata, ...] = (
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



# Additional public implementations and allow-listed model families. These are
# metadata records only: no implementation module is imported or executed here.
_SERIES_VALUES = AlgorithmPort(
    name="values", dtype="series[float]", description="Chronological finite numeric series"
)
_ARRAY_VALUES = AlgorithmPort(
    name="values", dtype="array[float]", description="Finite numeric observations"
)
_SERIES_RESULT = AlgorithmPort(
    name="values", dtype="series[float]", description="Algorithm output series"
)
_RECORD_RESULT = AlgorithmPort(
    name="result", dtype="record", description="Structured algorithm result"
)
_MODEL_RESULT = AlgorithmPort(
    name="model", dtype="fitted_model", description="Fitted algorithm model"
)
_MATRIX_RESULT = AlgorithmPort(
    name="matrix", dtype="matrix[float]", description="Numeric matrix result"
)
_FEATURES = AlgorithmPort(
    name="features", dtype="matrix[float]", description="Finite feature matrix"
)
_DATASET = AlgorithmPort(
    name="dataset", dtype="dataset", description="Validated, schema-conforming dataset"
)
_FUNDAMENTAL = AlgorithmPort(
    name="snapshot", dtype="fundamental_snapshot", description="Point-in-time fundamental snapshot"
)
_PORTFOLIO = AlgorithmPort(
    name="portfolio", dtype="portfolio", description="Portfolio with dated positions and prices"
)
_LEFT_SERIES = AlgorithmPort(name="left", dtype="series[float]", description="Left-hand series")
_RIGHT_SERIES = AlgorithmPort(name="right", dtype="series[float]", description="Right-hand series")
_HIGH = AlgorithmPort(name="high", dtype="series[float]", description="Chronological high prices", unit="price")
_LOW = AlgorithmPort(name="low", dtype="series[float]", description="Chronological low prices", unit="price")
_CLOSE = AlgorithmPort(name="close", dtype="series[float]", description="Chronological close prices", unit="price")
_VOLUME = AlgorithmPort(name="volume", dtype="series[float]", description="Chronological traded volume")
_FEATURE_TARGET = (
    _FEATURES,
    AlgorithmPort(name="target", dtype="array[float|str]", description="Target values aligned to feature rows"),
)
_INPUT_PORTS = {
    "series": (_SERIES_VALUES,),
    "array": (_ARRAY_VALUES,),
    "returns": (_RETURNS,),
    "prices": (_PRICE,),
    "pair": (_LEFT_SERIES, _RIGHT_SERIES),
    "ohlc": (_HIGH, _LOW, _CLOSE),
    "ohlcv": (_HIGH, _LOW, _CLOSE, _VOLUME),
    "fundamental": (_FUNDAMENTAL,),
    "portfolio": (_PORTFOLIO,),
    "features": (_FEATURES,),
    "feature_target": _FEATURE_TARGET,
    "dataset": (_DATASET,),
}
_OUTPUT_PORTS = {
    "scalar": _SCALAR,
    "series": _SERIES_RESULT,
    "record": _RECORD_RESULT,
    "model": _MODEL_RESULT,
    "matrix": _MATRIX_RESULT,
    "dataset": AlgorithmPort(name="dataset", dtype="dataset", description="Transformed dataset"),
    "paths": AlgorithmPort(name="paths", dtype="matrix[float]", description="Simulated paths by horizon"),
    "labels": AlgorithmPort(name="labels", dtype="array[int]", description="Per-observation labels"),
    "forecast": AlgorithmPort(name="forecast", dtype="forecast_distribution", description="Forecast distribution"),
    "prediction": AlgorithmPort(name="prediction", dtype="prediction", description="Model prediction output"),
}
def _extra_entry(spec: tuple[object, ...]) -> AlgorithmMetadata:
    (
        algorithm_id, name, category, summary, method, implementation_ref,
        input_kind, output_kind, tags,
    ) = spec[:9]
    parameter_spec = cast(
        tuple[str, str, str, str | int | float | bool | None] | None,
        spec[9] if len(spec) > 9 else None,
    )
    parameters: tuple[AlgorithmParameter, ...] = ()
    if parameter_spec is not None:
        p_name, p_dtype, p_description, p_default = parameter_spec
        parameters = (
            AlgorithmParameter(
                name=p_name, dtype=p_dtype, description=p_description, default=p_default
            ),
        )
    return _entry(
        str(algorithm_id), str(name), str(category), str(summary), str(method),
        _INPUT_PORTS[str(input_kind)], _OUTPUT_PORTS[str(output_kind)],
        implementation_ref=str(implementation_ref), parameters=parameters, tags=cast(tuple[str, ...], tags),
    )

_EXTRA_SPECS: tuple[tuple[object, ...], ...] = (
    # Return analytics and descriptive statistics.
    ("returns.wealth_index", "Wealth index", "returns", "Compound periodic returns into a cumulative wealth curve.", "W_t = W_0 × product(1 + r_t)", "quant.returns.performance.wealth_index", "returns", "series", ("returns", "performance", "compounding")),
    ("returns.drawdown_series", "Drawdown series", "risk", "Calculate the peak-to-trough drawdown at each observation.", "W_t / max(W_0…W_t) - 1", "quant.returns.performance.drawdown_series", "returns", "series", ("risk", "drawdown", "no-lookahead")),
    ("returns.max_drawdown", "Maximum drawdown", "risk", "Return the most severe peak-to-trough drawdown.", "min(drawdown_t)", "quant.returns.performance.max_drawdown", "returns", "scalar", ("risk", "drawdown")),
    ("returns.annualized_return", "Annualized return", "returns", "Annualize a periodic return sequence using an explicit frequency convention.", "geometric annualization using periods_per_year", "quant.returns.performance.annualized_return", "returns", "scalar", ("returns", "annualization")),
    ("returns.sharpe_ratio", "Sharpe ratio", "risk", "Estimate excess return per unit of return volatility.", "mean(excess returns) / std(excess returns) × annualization factor", "quant.returns.performance.sharpe_ratio", "returns", "scalar", ("risk", "risk-adjusted-return")),
    ("returns.downside_deviation", "Downside deviation", "risk", "Measure dispersion of returns below a declared minimum acceptable return.", "root mean squared downside shortfall", "quant.returns.performance.downside_deviation", "returns", "scalar", ("risk", "downside-risk")),
    ("returns.sortino_ratio", "Sortino ratio", "risk", "Estimate excess return relative to downside deviation.", "mean(excess returns) / downside deviation", "quant.returns.performance.sortino_ratio", "returns", "scalar", ("risk", "risk-adjusted-return")),
    ("returns.positive_period_ratio", "Positive-period ratio", "statistics", "Compute the fraction of periods with positive returns.", "count(r_t > 0) / n", "quant.returns.performance.positive_period_ratio", "returns", "scalar", ("returns", "descriptive")),
    ("statistics.quantile", "Empirical quantile", "statistics", "Compute a requested quantile from finite numeric observations.", "empirical quantile at probability p", "quant.statistics.descriptive.quantile", "array", "scalar", ("statistics", "distribution")),
    ("statistics.interquartile_range", "Interquartile range", "statistics", "Measure the spread between the first and third quartiles.", "Q3 - Q1", "quant.statistics.descriptive.interquartile_range", "array", "scalar", ("statistics", "robust")),
    ("statistics.median_absolute_deviation", "Median absolute deviation", "statistics", "Compute a robust measure of dispersion around the median.", "median(abs(x - median(x)))", "quant.statistics.descriptive.median_absolute_deviation", "array", "scalar", ("statistics", "robust")),
    ("statistics.skewness", "Skewness", "statistics", "Estimate distribution asymmetry from sample observations.", "standardized third central moment", "quant.statistics.descriptive.skewness", "array", "scalar", ("statistics", "distribution")),
    ("statistics.excess_kurtosis", "Excess kurtosis", "statistics", "Estimate tail weight relative to a Gaussian reference.", "standardized fourth central moment - 3", "quant.statistics.descriptive.excess_kurtosis", "array", "scalar", ("statistics", "distribution", "tails")),
    ("statistics.rankdata", "Rank data", "statistics", "Convert numeric observations to ranks using the implementation tie policy.", "rank(x_i)", "quant.statistics.dependence.rankdata", "array", "series", ("statistics", "rank")),
    ("statistics.spearman_correlation", "Spearman correlation", "statistics", "Measure monotonic association using ranked observations.", "correlation(rank(x), rank(y))", "quant.statistics.dependence.spearman_correlation", "pair", "scalar", ("statistics", "dependence", "rank")),
    ("statistics.autocorrelation", "Autocorrelation", "statistics", "Measure linear dependence between a series and its lagged values.", "corr(x_t, x_(t-lag))", "quant.statistics.dependence.autocorrelation", "series", "scalar", ("statistics", "time-series", "dependence")),
    ("math.weighted_mean", "Weighted mean", "statistics", "Compute a mean using aligned observation weights.", "sum(w_i x_i) / sum(w_i)", "quant.math.algebra.weighted_mean", "pair", "scalar", ("statistics", "aggregation")),
    ("math.geometric_mean", "Geometric mean", "statistics", "Compute the geometric mean of positive values.", "product(x_i) ** (1/n)", "quant.math.algebra.geometric_mean", "array", "scalar", ("statistics", "aggregation")),
    ("math.harmonic_mean", "Harmonic mean", "statistics", "Compute the harmonic mean of numeric observations.", "n / sum(1 / x_i)", "quant.math.algebra.harmonic_mean", "array", "scalar", ("statistics", "aggregation")),
    ("math.dot_product", "Dot product", "math", "Compute the dot product of aligned numeric vectors.", "sum(x_i y_i)", "quant.math.algebra.dot_product", "pair", "scalar", ("math", "linear-algebra")),
    ("math.compound_growth", "Compound growth", "math", "Project a value through a constant per-period growth rate.", "initial × (1 + rate) ** periods", "quant.math.algebra.compound_growth", "series", "scalar", ("math", "growth")),
    ("math.discount_factor", "Discount factor", "valuation", "Compute the discount factor for a period count and per-period rate.", "1 / (1 + r) ** n", "quant.math.financial.discount_factor", "series", "scalar", ("valuation", "discounting")),
    ("math.future_value", "Future value", "valuation", "Compound a present amount over a number of periods.", "PV × (1 + r) ** n", "quant.math.financial.future_value", "series", "scalar", ("valuation", "compounding")),
    ("math.annuity_present_value", "Annuity present value", "valuation", "Discount a level payment stream to present value.", "sum(payment / (1 + r) ** t)", "quant.math.financial.annuity_present_value", "series", "scalar", ("valuation", "annuity")),
    ("math.annuity_future_value", "Annuity future value", "valuation", "Compound a level payment stream to a future date.", "sum(payment × (1 + r) ** t)", "quant.math.financial.annuity_future_value", "series", "scalar", ("valuation", "annuity")),
    ("math.annuity_payment", "Annuity payment", "valuation", "Solve for the level payment consistent with a present value and rate.", "payment implied by annuity present-value equation", "quant.math.financial.annuity_payment", "series", "scalar", ("valuation", "annuity")),
    ("math.linear_interpolate", "Linear interpolation", "math", "Interpolate a value between two points on a straight line.", "y0 + (x-x0) × (y1-y0)/(x1-x0)", "quant.math.interpolation.linear_interpolate", "pair", "scalar", ("math", "interpolation")),
    ("math.linear_interpolate_series", "Series interpolation", "math", "Interpolate values across a declared set of target coordinates.", "piecewise linear interpolation", "quant.math.interpolation.linear_interpolate_series", "pair", "series", ("math", "interpolation")),
    ("math.golden_section_minimize", "Golden-section minimization", "optimization", "Minimize a unimodal scalar objective within a bounded interval.", "golden-section search", "quant.math.optimization.golden_section_minimize", "series", "record", ("optimization", "bounded-search")),
    ("math.normal_pdf", "Normal probability density", "probability", "Evaluate the normal probability density for a declared mean and standard deviation.", "Gaussian probability density", "quant.math.probability.normal_pdf", "series", "scalar", ("probability", "distribution")),
    ("math.binomial_pmf", "Binomial probability mass", "probability", "Evaluate the probability of k successes in n Bernoulli trials.", "comb(n,k) p^k (1-p)^(n-k)", "quant.math.probability.binomial_pmf", "series", "scalar", ("probability", "distribution")),
    ("math.population_variance", "Population variance", "statistics", "Compute variance treating observations as the full population.", "sum((x_i - mean(x))²) / n", "quant.math.statistics.population_variance", "array", "scalar", ("statistics", "dispersion")),
    ("math.sample_variance", "Sample variance", "statistics", "Compute unbiased sample variance using n-1 degrees of freedom.", "sum((x_i - mean(x))²) / (n-1)", "quant.math.statistics.sample_variance", "array", "scalar", ("statistics", "dispersion")),
    ("math.population_std", "Population standard deviation", "statistics", "Compute the population standard deviation.", "sqrt(population variance)", "quant.math.statistics.population_std", "array", "scalar", ("statistics", "dispersion")),
    ("math.sample_std", "Sample standard deviation", "statistics", "Compute the sample standard deviation.", "sqrt(sample variance)", "quant.math.statistics.sample_std", "array", "scalar", ("statistics", "dispersion")),
    ("math.sample_covariance", "Sample covariance", "statistics", "Compute covariance for aligned observations with sample correction.", "sample covariance(x, y)", "quant.math.statistics.sample_covariance", "pair", "scalar", ("statistics", "dependence")),
    ("math.correlation", "Pearson correlation", "statistics", "Compute linear correlation for aligned observations.", "cov(x, y) / (std(x) std(y))", "quant.math.statistics.correlation", "pair", "scalar", ("statistics", "dependence")),
    # Time-series estimation and diagnostics.
    ("timeseries.fit_ar", "Autoregressive model", "time_series", "Fit an autoregressive model to a chronological series.", "AR(p) parameter estimation", "quant.time_series.models.fit_ar", "series", "model", ("time-series", "forecast", "statistical")),
    ("timeseries.fit_ma", "Moving-average model", "time_series", "Fit a moving-average time-series model.", "MA(q) parameter estimation", "quant.time_series.models.fit_ma", "series", "model", ("time-series", "forecast", "statistical")),
    ("timeseries.fit_arma", "ARMA model", "time_series", "Fit an autoregressive moving-average model.", "ARMA(p,q) estimation", "quant.time_series.models.fit_arma", "series", "model", ("time-series", "forecast", "statistical")),
    ("timeseries.fit_sarima", "Seasonal ARIMA model", "time_series", "Fit a seasonal ARIMA model with explicit seasonal order.", "SARIMA(p,d,q)(P,D,Q,s)", "quant.time_series.models.fit_sarima", "series", "model", ("time-series", "forecast", "seasonal")),
    ("timeseries.fit_arimax", "ARIMAX model", "time_series", "Fit an ARIMA model with aligned exogenous predictors.", "ARIMA with exogenous regressors", "quant.time_series.models.fit_arimax", "feature_target", "model", ("time-series", "forecast", "exogenous")),
    ("timeseries.fit_exponential_smoothing", "Exponential smoothing", "time_series", "Fit an exponential-smoothing model to chronological observations.", "exponentially weighted level/trend estimation", "quant.time_series.models.fit_exponential_smoothing", "series", "model", ("time-series", "forecast", "smoothing")),
    ("timeseries.fit_holt", "Holt linear trend", "time_series", "Fit level-and-trend exponential smoothing.", "Holt level/trend recursion", "quant.time_series.models.fit_holt", "series", "model", ("time-series", "forecast", "trend")),
    ("timeseries.fit_holt_winters", "Holt-Winters seasonal model", "time_series", "Fit a level, trend and seasonal smoothing model.", "Holt-Winters state update equations", "quant.time_series.models.fit_holt_winters", "series", "model", ("time-series", "forecast", "seasonal")),
    ("timeseries.fit_var", "Vector autoregression", "time_series", "Fit a multivariate vector autoregression model.", "VAR(p) estimation", "quant.time_series.models.fit_var", "features", "model", ("time-series", "multivariate", "forecast")),
    ("timeseries.fit_vecm", "Vector error-correction model", "time_series", "Fit a multivariate error-correction model for cointegrated series.", "VECM estimation", "quant.time_series.models.fit_vecm", "features", "model", ("time-series", "cointegration", "multivariate")),
    ("timeseries.ljung_box_test", "Ljung-Box residual test", "time_series", "Test residuals for remaining autocorrelation up to a requested lag.", "Ljung-Box Q statistic", "quant.time_series.diagnostics.ljung_box_test", "series", "record", ("time-series", "diagnostics", "statistical-test")),
    ("timeseries.temporal_train_test_split", "Temporal train-test split", "time_series", "Split observations chronologically without random shuffling.", "prefix training segment, suffix test segment", "quant.time_series.validation.temporal_train_test_split", "series", "record", ("time-series", "validation", "no-lookahead")),
    # Cointegration, dependence, and regime detection.
    ("cointegration.adf_test", "Augmented Dickey-Fuller test", "cointegration", "Test a series for a unit root using augmented lagged differences.", "ADF regression and unit-root statistic", "quant.cointegration.models.adf_test", "series", "record", ("cointegration", "stationarity", "statistical-test")),
    ("cointegration.kpss_test", "KPSS stationarity test", "cointegration", "Test level- or trend-stationarity under the KPSS null.", "KPSS statistic", "quant.cointegration.models.kpss_test", "series", "record", ("cointegration", "stationarity", "statistical-test")),
    ("cointegration.phillips_perron_test", "Phillips-Perron test", "cointegration", "Test for a unit root with nonparametric serial-correlation correction.", "Phillips-Perron statistic", "quant.cointegration.models.phillips_perron_test", "series", "record", ("cointegration", "stationarity", "statistical-test")),
    ("cointegration.unit_root_diagnostics", "Unit-root diagnostics", "cointegration", "Combine complementary unit-root diagnostics for a chronological series.", "multi-test stationarity diagnostic", "quant.cointegration.models.unit_root_diagnostics", "series", "record", ("cointegration", "stationarity", "diagnostics")),
    ("cointegration.engle_granger_test", "Engle-Granger test", "cointegration", "Test a pair of series for cointegration using a residual-based procedure.", "Engle-Granger two-step test", "quant.cointegration.models.engle_granger_test", "pair", "record", ("cointegration", "pairs-trading", "statistical-test")),
    ("cointegration.johansen_test", "Johansen cointegration test", "cointegration", "Estimate cointegration rank in a multivariate system.", "Johansen trace and maximum-eigenvalue statistics", "quant.cointegration.models.johansen_test", "features", "record", ("cointegration", "multivariate", "statistical-test")),
    ("cointegration.fit_cointegrating_pair", "Cointegrating pair fit", "cointegration", "Estimate a pair relationship and its spread statistics.", "cointegrating regression and residual analysis", "quant.cointegration.models.fit_cointegrating_pair", "pair", "record", ("cointegration", "pairs-trading")),
    ("cointegration.spread_zscore", "Spread z-score", "cointegration", "Standardize a cointegrating spread for relative-value analysis.", "(spread - rolling mean) / rolling standard deviation", "quant.cointegration.models.spread_zscore", "series", "series", ("cointegration", "pairs-trading", "normalization")),
    ("cointegration.benjamini_hochberg", "Benjamini-Hochberg correction", "statistics", "Control the false-discovery rate across a family of p-values.", "step-up false-discovery-rate procedure", "quant.cointegration.models.benjamini_hochberg", "array", "series", ("statistics", "multiple-testing")),
    ("dependence.correlation_matrix", "Correlation matrix", "dependence", "Estimate the cross-series correlation matrix from a named panel.", "pairwise correlation matrix", "quant.dependence.models.correlation_matrix", "features", "matrix", ("dependence", "covariance")),
    ("dependence.covariance_matrix", "Covariance matrix", "dependence", "Estimate covariance across named return series.", "sample covariance matrix", "quant.dependence.models.covariance_matrix", "dataset", "matrix", ("dependence", "covariance")),
    ("dependence.oas_shrinkage_covariance", "OAS shrinkage covariance", "dependence", "Estimate a regularized covariance matrix using Oracle Approximating Shrinkage.", "OAS covariance shrinkage", "quant.dependence.models.oas_shrinkage_covariance", "dataset", "matrix", ("dependence", "covariance", "shrinkage")),
    ("dependence.rolling_dependency", "Rolling dependency", "dependence", "Estimate dependence measures over rolling windows.", "rolling dependence statistics", "quant.dependence.models.rolling_dependency", "dataset", "record", ("dependence", "rolling")),
    ("dependence.pca_factors", "Principal-component factors", "dependence", "Extract principal components from an aligned multivariate panel.", "eigendecomposition of covariance/correlation structure", "quant.dependence.models.pca_factors", "features", "record", ("dependence", "dimensionality-reduction")),
    # Option pricing and volatility.
    ("derivatives.black_scholes_price", "Black-Scholes option price", "derivatives", "Price a European option using the Black-Scholes framework.", "closed-form European option pricing", "quant.derivatives.models.black_scholes_price", "dataset", "scalar", ("derivatives", "option-pricing")),
    ("derivatives.binomial_tree_price", "Binomial-tree option price", "derivatives", "Price an option using a recombining binomial lattice.", "risk-neutral binomial backward induction", "quant.derivatives.models.binomial_tree_price", "dataset", "scalar", ("derivatives", "option-pricing", "lattice")),
    ("derivatives.trinomial_tree_price", "Trinomial-tree option price", "derivatives", "Price an option using a recombining trinomial lattice.", "risk-neutral trinomial backward induction", "quant.derivatives.models.trinomial_tree_price", "dataset", "scalar", ("derivatives", "option-pricing", "lattice")),
    ("derivatives.monte_carlo_price", "Monte Carlo option price", "derivatives", "Estimate an option value from simulated risk-neutral paths.", "discounted expectation of simulated payoffs", "quant.derivatives.models.monte_carlo_price", "dataset", "record", ("derivatives", "option-pricing", "simulation")),
    ("derivatives.black_scholes_greeks", "Black-Scholes Greeks", "derivatives", "Calculate option price sensitivities to model inputs.", "analytic partial derivatives of Black-Scholes value", "quant.derivatives.models.black_scholes_greeks", "dataset", "record", ("derivatives", "greeks", "sensitivity")),
    ("derivatives.implied_volatility", "Implied volatility", "derivatives", "Invert an option-pricing model to find volatility consistent with market price.", "bounded root finding on option price error", "quant.derivatives.models.implied_volatility", "dataset", "scalar", ("derivatives", "volatility", "calibration")),
    ("derivatives.build_volatility_surface", "Volatility surface", "derivatives", "Construct an implied-volatility surface over option strikes and maturities.", "interpolation across strike and maturity", "quant.derivatives.models.build_volatility_surface", "dataset", "record", ("derivatives", "volatility-surface")),
    ("volatility.rolling_volatility", "Rolling volatility", "volatility", "Estimate trailing return volatility using a rolling window.", "rolling standard deviation with annualization convention", "quant.volatility.models.rolling_volatility", "returns", "series", ("volatility", "rolling", "risk")),
    ("volatility.ewma_volatility", "EWMA volatility", "volatility", "Estimate volatility with exponentially weighted squared returns.", "EWMA variance recursion", "quant.volatility.models.ewma_volatility", "returns", "series", ("volatility", "ewma", "risk")),
    ("volatility.fit_arch", "ARCH model", "volatility", "Fit an ARCH conditional-variance model.", "ARCH(p) conditional variance likelihood", "quant.volatility.models.fit_arch", "returns", "model", ("volatility", "conditional-variance")),
    ("volatility.fit_garch", "GARCH model", "volatility", "Fit a GARCH conditional-variance model.", "GARCH(p,q) conditional variance likelihood", "quant.volatility.models.fit_garch", "returns", "model", ("volatility", "conditional-variance")),
    ("volatility.fit_egarch", "EGARCH model", "volatility", "Fit an exponential GARCH model for asymmetric volatility dynamics.", "EGARCH log-variance recursion", "quant.volatility.models.fit_egarch", "returns", "model", ("volatility", "asymmetry")),
    ("volatility.fit_gjr_garch", "GJR-GARCH model", "volatility", "Fit an asymmetric GARCH model with threshold effects.", "GJR-GARCH conditional variance recursion", "quant.volatility.models.fit_gjr_garch", "returns", "model", ("volatility", "asymmetry")),
    ("volatility.fit_stochastic_volatility", "Stochastic-volatility model", "volatility", "Fit a latent stochastic-volatility model.", "latent log-volatility state estimation", "quant.volatility.models.fit_stochastic_volatility", "returns", "model", ("volatility", "latent-state")),
    ("volatility.fit_dcc_garch", "DCC-GARCH model", "volatility", "Estimate dynamic conditional correlations jointly with univariate volatility models.", "DCC correlation recursion with GARCH marginals", "quant.volatility.models.fit_dcc_garch", "dataset", "model", ("volatility", "dependence", "multivariate")),
    ("stochastic.fit_ornstein_uhlenbeck", "Ornstein-Uhlenbeck fit", "stochastic_process", "Fit a mean-reverting Ornstein-Uhlenbeck process.", "discrete-time OU parameter estimation", "quant.stochastic.processes.fit_ornstein_uhlenbeck", "series", "model", ("stochastic-process", "mean-reversion")),
    ("stochastic.fit_mean_reversion", "Mean-reversion fit", "stochastic_process", "Fit a mean-reversion model and estimate its forecast behavior.", "mean-reversion state estimation", "quant.stochastic.processes.fit_mean_reversion", "series", "model", ("stochastic-process", "mean-reversion")),
    ("forecasting.forecast_return_distribution", "Return distribution forecast", "forecasting", "Estimate a predictive return distribution from historical observations.", "parametric/distributional forecast according to declared configuration", "quant.forecasting.models.forecast_return_distribution", "returns", "forecast", ("forecasting", "probability")),
    ("forecasting.forecast_return_distributions", "Batch return distribution forecast", "forecasting", "Generate return-distribution forecasts for multiple aligned series.", "batch application of configured distributional forecasting", "quant.forecasting.models.forecast_return_distributions", "dataset", "record", ("forecasting", "probability", "batch")),
    ("forecasting.simulate_return_paths", "Return-path simulation", "forecasting", "Simulate future return paths using historical return dynamics and optional regimes.", "bootstrap or regime-conditioned path simulation", "quant.forecasting.simulation.simulate_return_paths", "returns", "paths", ("forecasting", "simulation", "scenario")),
    ("forecasting.evaluate_forecast_calibration", "Forecast calibration evaluation", "forecasting", "Evaluate calibration of probabilistic forecasts against realized outcomes.", "empirical coverage and calibration diagnostics", "quant.forecasting.calibration.evaluate_forecast_calibration", "dataset", "record", ("forecasting", "calibration")),
    ("forecasting.walk_forward_evaluation", "Walk-forward forecast evaluation", "forecasting", "Evaluate a forecaster using expanding or rolling chronological splits.", "walk-forward evaluation without random shuffling", "quant.forecasting.calibration.walk_forward_forecast_evaluation", "dataset", "record", ("forecasting", "validation", "no-lookahead")),
    # Technical indicators.
    ("technical.sma", "Simple moving average", "technical", "Calculate the rolling arithmetic mean of close prices.", "SMA_t = mean(close[t-window+1:t])", "quant.technical.indicators.sma", "prices", "series", ("technical-analysis", "trend")),
    ("technical.ema", "Exponential moving average", "technical", "Calculate exponentially weighted moving average values.", "EMA recursive smoothing", "quant.technical.indicators.ema", "prices", "series", ("technical-analysis", "trend")),
    ("technical.wma", "Weighted moving average", "technical", "Calculate a linearly weighted moving average.", "weighted trailing mean", "quant.technical.indicators.wma", "prices", "series", ("technical-analysis", "trend")),
    ("technical.momentum", "Momentum indicator", "technical", "Measure price change over a configured lookback interval.", "close_t - close_(t-period)", "quant.technical.indicators.momentum", "prices", "series", ("technical-analysis", "momentum")),
    ("technical.rate_of_change", "Rate of change", "technical", "Calculate percentage change across a lookback interval.", "close_t / close_(t-period) - 1", "quant.technical.indicators.rate_of_change", "prices", "series", ("technical-analysis", "momentum")),
    ("technical.macd", "MACD", "technical", "Calculate moving-average convergence/divergence and its signal series.", "EMA_fast - EMA_slow and signal EMA", "quant.technical.indicators.macd", "prices", "record", ("technical-analysis", "momentum")),
    ("technical.atr", "Average true range", "technical", "Estimate volatility from high, low and previous close prices.", "smoothed true range", "quant.technical.indicators.atr", "ohlc", "series", ("technical-analysis", "volatility")),
    ("technical.bollinger_bands", "Bollinger Bands", "technical", "Calculate moving-average bands based on rolling dispersion.", "SMA ± k × rolling standard deviation", "quant.technical.indicators.bollinger_bands", "prices", "record", ("technical-analysis", "volatility")),
    ("technical.stochastic_oscillator", "Stochastic oscillator", "technical", "Compare close prices with the trailing high-low range.", "%K and smoothed %D", "quant.technical.indicators.stochastic_oscillator", "ohlc", "record", ("technical-analysis", "momentum")),
    ("technical.williams_r", "Williams %R", "technical", "Measure close relative to the trailing high-low range.", "-100 × (highest high - close)/(highest high - lowest low)", "quant.technical.indicators.williams_r", "ohlc", "series", ("technical-analysis", "momentum")),
    ("technical.adx_dmi", "ADX and directional movement", "technical", "Estimate trend strength and directional movement components.", "smoothed directional movement and DX", "quant.technical.indicators.adx_dmi", "ohlc", "record", ("technical-analysis", "trend")),
    ("technical.donchian_channels", "Donchian channels", "technical", "Calculate trailing extrema bands from high and low prices.", "rolling highest high and lowest low", "quant.technical.indicators.donchian_channels", "ohlc", "record", ("technical-analysis", "breakout")),
    ("technical.vwap", "Volume-weighted average price", "technical", "Calculate a volume-weighted average price series.", "cumulative sum(typical price × volume) / cumulative volume", "quant.technical.indicators.vwap", "ohlcv", "series", ("technical-analysis", "volume")),
    ("technical.volume_change", "Volume change", "technical", "Measure change in traded volume over a lookback interval.", "volume_t / volume_(t-period) - 1", "quant.technical.indicators.volume_change", "series", "series", ("technical-analysis", "volume")),
    # Fundamental analysis and valuation.
    ("fundamentals.growth_rate", "Fundamental growth rate", "fundamentals", "Calculate growth between current and prior comparable values.", "current / previous - 1", "quant.fundamentals.growth.growth_rate", "pair", "scalar", ("fundamentals", "growth")),
    ("fundamentals.roe", "Return on equity", "fundamentals", "Compute return on equity from a point-in-time fundamental snapshot.", "net income / shareholders' equity", "quant.fundamentals.ratios.roe", "fundamental", "scalar", ("fundamentals", "profitability")),
    ("fundamentals.roa", "Return on assets", "fundamentals", "Compute return on assets.", "net income / total assets", "quant.fundamentals.ratios.roa", "fundamental", "scalar", ("fundamentals", "profitability")),
    ("fundamentals.roic", "Return on invested capital", "fundamentals", "Compute after-tax operating return on invested capital.", "NOPAT / invested capital", "quant.fundamentals.ratios.roic", "fundamental", "scalar", ("fundamentals", "profitability")),
    ("fundamentals.roce", "Return on capital employed", "fundamentals", "Compute return on capital employed.", "EBIT / capital employed", "quant.fundamentals.ratios.roce", "fundamental", "scalar", ("fundamentals", "profitability")),
    ("fundamentals.ebitda_margin", "EBITDA margin", "fundamentals", "Compute EBITDA relative to revenue.", "EBITDA / revenue", "quant.fundamentals.ratios.ebitda_margin", "fundamental", "scalar", ("fundamentals", "margin")),
    ("fundamentals.ebit_margin", "EBIT margin", "fundamentals", "Compute operating earnings relative to revenue.", "EBIT / revenue", "quant.fundamentals.ratios.ebit_margin", "fundamental", "scalar", ("fundamentals", "margin")),
    ("fundamentals.net_margin", "Net margin", "fundamentals", "Compute net income relative to revenue.", "net income / revenue", "quant.fundamentals.ratios.net_margin", "fundamental", "scalar", ("fundamentals", "margin")),
    ("fundamentals.free_cash_flow_margin", "Free-cash-flow margin", "fundamentals", "Compute free cash flow relative to revenue.", "free cash flow / revenue", "quant.fundamentals.ratios.free_cash_flow_margin", "fundamental", "scalar", ("fundamentals", "cash-flow")),
    ("fundamentals.cash_conversion_ratio", "Cash conversion ratio", "fundamentals", "Measure cash conversion from accounting earnings.", "operating cash flow / net income (per implementation contract)", "quant.fundamentals.ratios.cash_conversion_ratio", "fundamental", "scalar", ("fundamentals", "cash-flow")),
    ("fundamentals.payout_ratio", "Dividend payout ratio", "fundamentals", "Compute dividend payout relative to earnings.", "dividends / earnings", "quant.fundamentals.ratios.payout_ratio", "fundamental", "scalar", ("fundamentals", "dividends")),
    ("fundamentals.retention_ratio", "Earnings retention ratio", "fundamentals", "Compute the fraction of earnings retained after dividends.", "1 - payout ratio", "quant.fundamentals.ratios.retention_ratio", "fundamental", "scalar", ("fundamentals", "dividends")),
    ("fundamentals.debt_to_equity", "Debt-to-equity ratio", "fundamentals", "Measure financial leverage relative to equity.", "debt / equity", "quant.fundamentals.ratios.debt_to_equity", "fundamental", "scalar", ("fundamentals", "leverage")),
    ("fundamentals.debt_to_assets", "Debt-to-assets ratio", "fundamentals", "Measure liabilities/debt relative to assets per the financial contract.", "debt / assets", "quant.fundamentals.ratios.debt_to_assets", "fundamental", "scalar", ("fundamentals", "leverage")),
    ("fundamentals.interest_coverage", "Interest coverage", "fundamentals", "Measure operating earnings available to cover interest expense.", "EBIT / interest expense", "quant.fundamentals.ratios.interest_coverage", "fundamental", "scalar", ("fundamentals", "leverage")),
    ("valuation.peer_multiple_mean", "Peer multiple mean", "valuation", "Summarize peer valuation multiples using the arithmetic mean.", "sum(peer multiples) / number of peers", "quant.valuation.comparables.peer_multiple_mean", "array", "scalar", ("valuation", "comparables")),
    ("valuation.peer_multiple_median", "Peer multiple median", "valuation", "Summarize peer valuation multiples using the median.", "median(peer multiples)", "quant.valuation.comparables.peer_multiple_median", "array", "scalar", ("valuation", "comparables")),
    ("valuation.comparable_valuation_result", "Comparable valuation", "valuation", "Estimate a valuation output using selected peer multiples.", "peer multiple × financial metric with explicit assumptions", "quant.valuation.comparables.comparable_valuation_result", "dataset", "record", ("valuation", "comparables")),
    ("valuation.dcf_result", "Discounted cash-flow valuation", "valuation", "Estimate enterprise or equity value from projected cash flows and discount assumptions.", "discounted projected cash flows plus terminal value", "quant.valuation.dcf.dcf_result", "dataset", "record", ("valuation", "dcf")),
    ("valuation.dcf_sensitivity", "DCF sensitivity analysis", "valuation", "Evaluate how DCF results vary across discount and growth assumptions.", "grid of DCF valuation outcomes", "quant.valuation.dcf.dcf_sensitivity", "dataset", "matrix", ("valuation", "dcf", "sensitivity")),
    ("valuation.gordon_growth_value", "Gordon growth valuation", "valuation", "Value a growing perpetuity under a constant growth assumption.", "D1 / (required return - growth)", "quant.valuation.ddm.gordon_growth_value", "dataset", "scalar", ("valuation", "dividends")),
    ("valuation.ddm_value", "Dividend discount valuation", "valuation", "Value equity from a dividend forecast and discount rate.", "sum(discounted forecast dividends) plus terminal value", "quant.valuation.ddm.ddm_value", "dataset", "scalar", ("valuation", "dividends")),
    ("valuation.pe_ratio", "Price-to-earnings ratio", "valuation", "Compute price relative to earnings per share.", "price / earnings per share", "quant.valuation.multiples.pe_ratio", "pair", "scalar", ("valuation", "multiples")),
    ("valuation.ps_ratio", "Price-to-sales ratio", "valuation", "Compute price relative to revenue per share.", "price / revenue per share", "quant.valuation.multiples.ps_ratio", "pair", "scalar", ("valuation", "multiples")),
    ("valuation.pb_ratio", "Price-to-book ratio", "valuation", "Compute price relative to book value per share.", "price / book value per share", "quant.valuation.multiples.pb_ratio", "pair", "scalar", ("valuation", "multiples")),
    ("valuation.ev_to_ebitda", "EV/EBITDA multiple", "valuation", "Compute enterprise value relative to EBITDA.", "enterprise value / EBITDA", "quant.valuation.multiples.ev_to_ebitda", "dataset", "scalar", ("valuation", "multiples")),
    ("valuation.ev_to_fcf", "EV/FCF multiple", "valuation", "Compute enterprise value relative to free cash flow.", "enterprise value / free cash flow", "quant.valuation.multiples.ev_to_fcf", "dataset", "scalar", ("valuation", "multiples")),
    ("valuation.earnings_yield", "Earnings yield", "valuation", "Compute earnings per share relative to price.", "earnings per share / price", "quant.valuation.multiples.earnings_yield", "pair", "scalar", ("valuation", "multiples")),
    ("valuation.nav_value", "Net asset value", "valuation", "Compute assets less liabilities.", "assets - liabilities", "quant.valuation.nav.nav_value", "dataset", "scalar", ("valuation", "asset-based")),
    ("valuation.residual_income_value", "Residual-income valuation", "valuation", "Estimate equity value from book value and forecast residual income.", "book value plus discounted residual income", "quant.valuation.residual_income.residual_income_value", "dataset", "scalar", ("valuation", "residual-income")),
    ("valuation.implied_terminal_growth", "Reverse DCF terminal growth", "valuation", "Infer terminal growth consistent with observed enterprise value under DCF assumptions.", "root solve for terminal growth matching observed value", "quant.valuation.reverse_dcf.implied_terminal_growth", "dataset", "scalar", ("valuation", "reverse-dcf")),
    ("valuation.sotp_equity_value", "Sum-of-the-parts valuation", "valuation", "Aggregate segment values with declared net-debt and adjustment assumptions.", "sum(segment values) - net debt + adjustments", "quant.valuation.sotp.sotp_equity_value", "dataset", "scalar", ("valuation", "sum-of-the-parts")),
    # Portfolio risk and tail-risk estimation.
    ("risk.beta", "Market beta", "risk", "Estimate sensitivity of asset returns to benchmark returns.", "cov(asset, benchmark) / var(benchmark)", "risk.metrics.performance.beta", "pair", "scalar", ("risk", "market-risk")),
    ("risk.treynor_ratio", "Treynor ratio", "risk", "Measure excess return per unit of systematic market risk.", "annualized excess return / beta", "risk.metrics.performance.treynor_ratio", "pair", "scalar", ("risk", "risk-adjusted-return")),
    ("risk.calmar_ratio", "Calmar ratio", "risk", "Compare annualized return with maximum drawdown magnitude.", "annualized return / abs(max drawdown)", "risk.metrics.performance.calmar_ratio", "returns", "scalar", ("risk", "risk-adjusted-return")),
    ("risk.omega_ratio", "Omega ratio", "risk", "Compare probability-weighted gains and losses around a threshold.", "sum(positive excesses) / abs(sum(negative excesses))", "risk.metrics.performance.omega_ratio", "returns", "scalar", ("risk", "distribution")),
    ("risk.information_ratio", "Information ratio", "risk", "Measure active return per unit of tracking error.", "mean(active returns) / std(active returns)", "risk.metrics.performance.information_ratio", "pair", "scalar", ("risk", "benchmark-relative")),
    ("risk.portfolio_exposure_metrics", "Portfolio exposure metrics", "risk", "Summarize portfolio exposure and concentration metrics.", "position-level exposure aggregation", "risk.metrics.portfolio.portfolio_exposure_metrics", "portfolio", "record", ("risk", "portfolio")),
    ("risk.liquidity_risk_metrics", "Liquidity risk metrics", "risk", "Summarize liquidity exposure using available portfolio and market data.", "liquidity metrics from portfolio inputs", "risk.metrics.portfolio.liquidity_risk_metrics", "portfolio", "record", ("risk", "liquidity")),
    ("risk.turnover_risk_metrics", "Turnover risk metrics", "risk", "Estimate portfolio turnover-related risk metrics.", "turnover aggregation from position changes", "risk.metrics.portfolio.turnover_risk_metrics", "portfolio", "record", ("risk", "turnover")),
    ("risk.historical_var_cvar", "Historical VaR and CVaR", "tail_risk", "Estimate tail losses from the empirical return distribution.", "historical quantile and conditional tail mean", "risk.tail.estimators.historical_var_cvar", "returns", "record", ("risk", "tail-risk", "var", "cvar")),
    ("risk.parametric_var_cvar", "Parametric VaR and CVaR", "tail_risk", "Estimate tail losses under a declared parametric return distribution.", "parametric quantile and expected tail loss", "risk.tail.estimators.parametric_var_cvar", "returns", "record", ("risk", "tail-risk", "var", "cvar")),
    ("risk.monte_carlo_var_cvar", "Monte Carlo VaR and CVaR", "tail_risk", "Estimate tail losses from simulated portfolio or return outcomes.", "Monte Carlo quantile and conditional tail mean", "risk.tail.estimators.monte_carlo_var_cvar", "returns", "record", ("risk", "tail-risk", "simulation")),
    ("risk.evt_var_cvar", "Extreme-value VaR and CVaR", "tail_risk", "Estimate tail risk using an extreme-value model.", "extreme-value tail extrapolation", "risk.tail.estimators.evt_var_cvar", "returns", "record", ("risk", "tail-risk", "extreme-value")),
    ("risk.evaluate_stress_scenario", "Stress scenario evaluation", "risk", "Evaluate a declared stress scenario against portfolio holdings.", "scenario shock propagation and portfolio impact", "risk.tail.scenarios.evaluate_stress_scenario", "portfolio", "record", ("risk", "stress-testing")),
    ("risk.stress_test_portfolio", "Portfolio stress testing", "risk", "Apply a set of stress scenarios to portfolio exposures.", "portfolio repricing under scenario set", "risk.tail.scenarios.stress_test_portfolio", "portfolio", "record", ("risk", "stress-testing")),
    ("risk.evaluate_constraints", "Portfolio constraint evaluation", "risk", "Evaluate portfolio decisions against explicit risk constraints.", "constraint-by-constraint feasibility checks", "risk.constraints.evaluator.evaluate_constraints", "portfolio", "record", ("risk", "constraints")),
    ("risk.evaluate_portfolio_risk", "Portfolio risk evaluation", "risk", "Evaluate portfolio risk measures and constraints together.", "risk metric aggregation plus constraint assessment", "risk.constraints.evaluator.evaluate_portfolio_risk", "portfolio", "record", ("risk", "constraints", "portfolio")),
    # Explicit ML model families; selected via existing allow-lists, not arbitrary imports.
    ("ml.linear_regression", "Linear regression", "machine_learning.regression", "Fit the allow-listed linear regression estimator.", "ordinary least-squares estimator", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "linear_regression")),
    ("ml.ridge", "Ridge regression", "machine_learning.regression", "Fit ridge regression with L2 regularization.", "least squares plus L2 penalty", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "regularization"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "ridge")),
    ("ml.lasso", "Lasso regression", "machine_learning.regression", "Fit Lasso regression with L1 regularization.", "least squares plus L1 penalty", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "regularization"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "lasso")),
    ("ml.elastic_net", "Elastic-net regression", "machine_learning.regression", "Fit regression with combined L1 and L2 regularization.", "least squares plus elastic-net penalty", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "regularization"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "elastic_net")),
    ("ml.svr", "Support-vector regression", "machine_learning.regression", "Fit a support-vector regression model.", "epsilon-support-vector regression", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "kernel"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "svr")),
    ("ml.knn_regressor", "K-nearest-neighbors regression", "machine_learning.regression", "Fit a KNN regressor using neighboring training examples.", "neighbor distance-weighted regression according to configuration", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "neighbors"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "knn_regressor")),
    ("ml.decision_tree_regressor", "Decision-tree regression", "machine_learning.regression", "Fit a decision-tree regression model.", "recursive feature-space partitioning", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "tree"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "decision_tree_regressor")),
    ("ml.random_forest_regressor", "Random-forest regression", "machine_learning.regression", "Fit a randomized ensemble of regression trees.", "bagged decision trees with feature subsampling", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "ensemble"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "random_forest_regressor")),
    ("ml.gradient_boosting_regressor", "Gradient-boosting regression", "machine_learning.regression", "Fit a boosted-tree regression model with validation-aware iteration selection.", "additive gradient-boosted decision trees", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "regression", "boosting"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "gradient_boosting_regressor")),
    ("ml.logistic_regression", "Logistic regression", "machine_learning.classification", "Fit a logistic-regression classifier.", "logistic likelihood optimization", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "classification"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "logistic_regression")),
    ("ml.svc", "Support-vector classification", "machine_learning.classification", "Fit a support-vector classifier.", "maximum-margin classification with configured kernel", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "classification", "kernel"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "svc")),
    ("ml.knn_classifier", "K-nearest-neighbors classification", "machine_learning.classification", "Classify observations using nearest training examples.", "neighbor-vote classification", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "classification", "neighbors"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "knn_classifier")),
    ("ml.decision_tree_classifier", "Decision-tree classification", "machine_learning.classification", "Fit a classification decision tree.", "recursive impurity-reducing partitioning", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "classification", "tree"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "decision_tree_classifier")),
    ("ml.random_forest_classifier", "Random-forest classification", "machine_learning.classification", "Fit a randomized ensemble of classification trees.", "bagged classification trees with feature subsampling", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "classification", "ensemble"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "random_forest_classifier")),
    ("ml.gradient_boosting_classifier", "Gradient-boosting classification", "machine_learning.classification", "Fit a gradient-boosted classification ensemble.", "additive gradient-boosted decision trees", "ml.classical.ClassicalMLModel.fit", "feature_target", "model", ("machine-learning", "classification", "boosting"), ("algorithm", "str", "Allow-listed estimator selected by the model contract", "gradient_boosting_classifier")),
    ("ml.unsupervised_kmeans", "K-means clustering", "machine_learning.clustering", "Partition observations into centroid-based clusters.", "minimize within-cluster squared distance", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "labels", ("machine-learning", "clustering"), ("algorithm", "str", "Allow-listed unsupervised estimator", "kmeans")),
    ("ml.unsupervised_agglomerative", "Agglomerative clustering", "machine_learning.clustering", "Build hierarchical clusters by recursively merging nearby groups.", "agglomerative hierarchical linkage", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "labels", ("machine-learning", "clustering"), ("algorithm", "str", "Allow-listed unsupervised estimator", "agglomerative_clustering")),
    ("ml.unsupervised_dbscan", "DBSCAN clustering", "machine_learning.clustering", "Find density-connected clusters and noise observations.", "density reachability with eps and minimum-sample thresholds", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "labels", ("machine-learning", "clustering", "density"), ("algorithm", "str", "Allow-listed unsupervised estimator", "dbscan")),
    ("ml.unsupervised_gaussian_mixture", "Gaussian-mixture clustering", "machine_learning.clustering", "Model clusters as a mixture of Gaussian components.", "Gaussian-mixture likelihood and expectation-maximization", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "labels", ("machine-learning", "clustering", "probabilistic"), ("algorithm", "str", "Allow-listed unsupervised estimator", "gaussian_mixture")),
    ("ml.unsupervised_pca", "Principal component analysis", "machine_learning.dimensionality_reduction", "Project observations onto leading linear components.", "SVD/eigendecomposition principal components", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "matrix", ("machine-learning", "dimensionality-reduction"), ("algorithm", "str", "Allow-listed unsupervised estimator", "pca")),
    ("ml.unsupervised_kernel_pca", "Kernel PCA", "machine_learning.dimensionality_reduction", "Perform nonlinear dimensionality reduction with a kernel matrix.", "kernel principal component analysis", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "matrix", ("machine-learning", "dimensionality-reduction", "kernel"), ("algorithm", "str", "Allow-listed unsupervised estimator", "kernel_pca")),
    ("ml.unsupervised_fast_ica", "Fast independent component analysis", "machine_learning.dimensionality_reduction", "Separate statistically independent latent components.", "fixed-point independent component analysis", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "matrix", ("machine-learning", "dimensionality-reduction"), ("algorithm", "str", "Allow-listed unsupervised estimator", "fast_ica")),
    ("ml.unsupervised_isolation_forest", "Isolation forest", "machine_learning.anomaly_detection", "Identify anomalies through randomized isolation trees.", "isolation path-length anomaly score", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "labels", ("machine-learning", "anomaly-detection"), ("algorithm", "str", "Allow-listed unsupervised estimator", "isolation_forest")),
    ("ml.unsupervised_local_outlier_factor", "Local outlier factor", "machine_learning.anomaly_detection", "Score local-density deviations to detect outliers.", "local reachability density ratio", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "labels", ("machine-learning", "anomaly-detection"), ("algorithm", "str", "Allow-listed unsupervised estimator", "local_outlier_factor")),
    ("ml.unsupervised_elliptic_envelope", "Elliptic-envelope anomaly detection", "machine_learning.anomaly_detection", "Detect outliers using a robust covariance envelope.", "Mahalanobis-distance based outlier detection", "ml.unsupervised.UnsupervisedMLModel.fit", "features", "labels", ("machine-learning", "anomaly-detection"), ("algorithm", "str", "Allow-listed unsupervised estimator", "elliptic_envelope")),
    ("ml.deep_attention", "Attention network", "machine_learning.deep_learning", "Train the allow-listed attention architecture for supervised sequence learning.", "attention-based neural network", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "attention"), ("architecture", "str", "Architecture selected by the deep-learning contract", "attention")),
    ("ml.deep_gru", "GRU network", "machine_learning.deep_learning", "Train a gated recurrent-unit network.", "GRU recurrent neural network", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "recurrent"), ("architecture", "str", "Architecture selected by the deep-learning contract", "gru")),
    ("ml.deep_lstm", "LSTM network", "machine_learning.deep_learning", "Train a long short-term-memory network.", "LSTM recurrent neural network", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "recurrent"), ("architecture", "str", "Architecture selected by the deep-learning contract", "lstm")),
    ("ml.deep_mlp", "Multilayer perceptron", "machine_learning.deep_learning", "Train a dense multilayer perceptron.", "feed-forward fully connected neural network", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "dense"), ("architecture", "str", "Architecture selected by the deep-learning contract", "mlp")),
    ("ml.deep_tcn", "Temporal convolutional network", "machine_learning.deep_learning", "Train a temporal convolutional network.", "causal/dilated temporal convolutions according to architecture contract", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "convolution"), ("architecture", "str", "Architecture selected by the deep-learning contract", "tcn")),
    ("ml.deep_transformer", "Transformer network", "machine_learning.deep_learning", "Train a transformer model for sequential inputs.", "self-attention plus position-aware feed-forward layers", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "transformer"), ("architecture", "str", "Architecture selected by the deep-learning contract", "transformer")),
    ("ml.deep_temporal_fusion_transformer", "Temporal fusion transformer", "machine_learning.deep_learning", "Train the temporal-fusion-transformer architecture exposed by the allow-list.", "gated variable selection and temporal attention", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "forecasting"), ("architecture", "str", "Architecture selected by the deep-learning contract", "temporal_fusion_transformer")),
    ("ml.deep_autoencoder", "Dense autoencoder", "machine_learning.reconstruction", "Train an autoencoder to reconstruct input features.", "encoder-decoder reconstruction objective", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "reconstruction"), ("architecture", "str", "Architecture selected by the deep-learning contract", "autoencoder")),
    ("ml.deep_vae", "Variational autoencoder", "machine_learning.reconstruction", "Train a variational autoencoder with latent-variable regularization.", "reconstruction loss plus KL divergence", "ml.deep_learning.DeepLearningModel.fit", "dataset", "model", ("machine-learning", "deep-learning", "reconstruction", "probabilistic"), ("architecture", "str", "Architecture selected by the deep-learning contract", "vae")),
    # F32 feature operations are registered separately because they have different contracts.
    ("features.identity", "Identity feature transform", "feature_engineering", "Pass through a declared source feature without changing its values.", "identity transform", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "identity"), ("operation", "str", "F32 allow-listed feature operation", "identity")),
    ("features.lag", "Lag feature", "feature_engineering", "Create a lagged version of a source feature without using future observations.", "x_(t-lag)", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "lag", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "lag")),
    ("features.pct_change", "Percentage-change feature", "feature_engineering", "Create a period percentage-change feature.", "x_t / x_(t-1) - 1", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "returns"), ("operation", "str", "F32 allow-listed feature operation", "pct_change")),
    ("features.log_return", "Log-return feature", "feature_engineering", "Create a logarithmic change feature for positive inputs.", "ln(x_t / x_(t-1))", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "returns"), ("operation", "str", "F32 allow-listed feature operation", "log_return")),
    ("features.rolling_mean", "Rolling-mean feature", "feature_engineering", "Create a trailing rolling-mean feature.", "mean(x[t-window+1:t])", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "rolling", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "rolling_mean")),
    ("features.rolling_std", "Rolling-standard-deviation feature", "feature_engineering", "Create a trailing rolling-dispersion feature.", "std(x[t-window+1:t])", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "rolling", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "rolling_std")),
    ("features.rolling_min", "Rolling-minimum feature", "feature_engineering", "Create a trailing-window minimum feature.", "min(x[t-window+1:t])", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "rolling", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "rolling_min")),
    ("features.rolling_max", "Rolling-maximum feature", "feature_engineering", "Create a trailing-window maximum feature.", "max(x[t-window+1:t])", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "rolling", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "rolling_max")),
    ("features.rolling_sum", "Rolling-sum feature", "feature_engineering", "Create a trailing-window sum feature.", "sum(x[t-window+1:t])", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "rolling", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "rolling_sum")),
    ("features.rolling_zscore", "Rolling-z-score feature", "feature_engineering", "Standardize a feature against trailing-window statistics.", "(x_t - rolling mean) / rolling std", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "normalization", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "rolling_zscore")),
    ("features.ewm_mean", "Exponentially weighted mean feature", "feature_engineering", "Create an exponentially weighted feature using prior and current observations.", "EWMA recursion", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "smoothing", "no-lookahead"), ("operation", "str", "F32 allow-listed feature operation", "ewm_mean")),
    ("features.interaction", "Interaction feature", "feature_engineering", "Create an interaction between declared source features.", "x_i × x_j", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "interaction"), ("operation", "str", "F32 allow-listed feature operation", "interaction")),
    ("features.ratio", "Ratio feature", "feature_engineering", "Create a ratio feature with explicit denominator handling.", "x_i / x_j", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "ratio"), ("operation", "str", "F32 allow-listed feature operation", "ratio")),
    ("features.cross_sectional_rank", "Cross-sectional rank feature", "feature_engineering", "Rank eligible assets within a point-in-time cross-section.", "rank(x_i within timestamp)", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "cross-sectional", "point-in-time"), ("operation", "str", "F32 allow-listed feature operation", "cross_sectional_rank")),
    ("features.cross_sectional_zscore", "Cross-sectional z-score feature", "feature_engineering", "Standardize eligible assets within a point-in-time cross-section.", "cross-sectional z-score within timestamp", "ml.features.engineering.FeatureEngineer.transform", "dataset", "dataset", ("feature-engineering", "cross-sectional", "point-in-time"), ("operation", "str", "F32 allow-listed feature operation", "cross_sectional_zscore")),
)

_EXTRA_ALGORITHMS: tuple[AlgorithmMetadata, ...] = tuple(
    _extra_entry(spec) for spec in _EXTRA_SPECS
)
BUILTIN_ALGORITHMS: tuple[AlgorithmMetadata, ...] = _SEED_ALGORITHMS + _EXTRA_ALGORITHMS


def builtin_algorithms() -> tuple[AlgorithmMetadata, ...]:
    """Return seed metadata without importing implementation modules."""
    return BUILTIN_ALGORITHMS


__all__ = ["BUILTIN_ALGORITHMS", "builtin_algorithms"]
