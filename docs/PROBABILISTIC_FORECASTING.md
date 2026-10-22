# F27 — Probabilistic Forecasting Engine

## Purpose

F27 turns historical returns into a reproducible predictive distribution instead of reporting only a point estimate. It provides cumulative-return quantiles, central predictive intervals, a probability of loss, Monte Carlo paths, interval calibration and strictly chronological walk-forward evaluation.

## Public API

From `quant.forecasting`:

- `simulate_return_paths(...)`: returns simulated per-period simple-return paths.
- `forecast_return_distribution(...)`: estimates one horizon's cumulative simple-return distribution.
- `forecast_return_distributions(...)`: repeats the forecast at several unique horizons; defaults are 1, 5, 10 and 20 periods.
- `evaluate_forecast_calibration(...)`: compares stored forecasts with realized cumulative returns.
- `walk_forward_forecast_evaluation(...)`: uses an expanding training window and evaluates only subsequent outcomes.

Contracts live in `core/contracts/forecasting.py`. Results include the training sample size, horizon, simulation count, random seed, method, quantiles and intervals.

## Methods and assumptions

### IID empirical bootstrap

The default method samples each future period with replacement from the supplied training returns. Each simulated path is compounded as `product(1 + r_t) - 1`, preserving simple-return compounding. This method captures the empirical marginal return distribution but **does not preserve serial dependence or volatility clustering**.

### Markov regime bootstrap

When both `regime_labels` and `transition_matrix` are provided, each simulated state follows the supplied row-stochastic transition matrix and each period's return is sampled from the historical return pool for that state. This path conditions on the final observed state and can use F26 outputs.

Regime labels and transition probabilities must be estimated from training data only. A label sequence produced by fitting a regime model on the complete sample may contain look-ahead information and is not acceptable for a strict out-of-sample test. Every state must have at least two training observations. The method is conditional on empirical within-state distributions; it is not a full generative model of market prices.

## Calibration and walk-forward validation

The calibration function reports:

- empirical coverage of a selected central interval;
- mean interval width and the Winkler-style interval score;
- Brier score for the event that the cumulative return is negative;
- MAE and MSE of the forecast expected return.

`walk_forward_forecast_evaluation` forecasts at each origin using only the expanding prefix strictly before that origin. The realized value compounds the next `horizon` returns. The returned object preserves origins, individual forecasts, realized returns and aggregate scores. For overlapping multi-period horizons, observations are not independent; interpret aggregate uncertainty accordingly.

## Validations and limitations

- At least 20 finite historical simple returns are required; each return must be greater than -100% and sample variance must be positive.
- Horizons must be positive integers; simulations must be at least 100; seeds must be non-negative and are recorded.
- Probability levels must be unique, ascending and strictly between zero and one.
- No missing data is silently imputed.
- All intervals are empirical simulation intervals, not guaranteed coverage statements.
- A high probability of loss is not a trading signal. F27 does not place orders and cannot override the F21 Risk Engine or portfolio risk controls.
- Bootstrap quality depends on the representativeness of the training window. Historical outputs do not guarantee future performance.

## Example

```python
from quant.forecasting import forecast_return_distributions

forecasts = forecast_return_distributions(
    historical_simple_returns,
    horizons=(1, 5, 10, 20),
    simulations=10_000,
    random_seed=27,
)

for forecast in forecasts:
    interval_90 = next(item for item in forecast.intervals if item.coverage == 0.9)
    print(
        forecast.horizon,
        forecast.expected_return,
        forecast.probability_of_loss,
        interval_90.lower_return,
        interval_90.upper_return,
    )
```

For model assessment, generate forecasts strictly out of sample and then pass each forecast and its matching realized cumulative return to `evaluate_forecast_calibration`, or use `walk_forward_forecast_evaluation` to create both chronologically.
