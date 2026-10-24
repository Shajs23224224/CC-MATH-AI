# F40 — Walk-Forward / Out-of-Sample Validation

## Purpose

F40 evaluates a strategy over chronological, out-of-sample (OOS) windows using the deterministic F39 backtest engine. It provides repeatable temporal splits, fold-level diagnostics, cross-window dispersion and optional performance summaries by externally supplied market regime.

This framework evaluates software and research protocols; it does not establish that a strategy is profitable or will remain profitable.

## Temporal split modes

- **rolling**: training length remains fixed and the window advances through time.
- **expanding**: the training prefix starts at the first observation and grows after each OOS window.
- **anchored**: training expands from an explicit, fixed anchor index. This supports excluding an initial warm-up or using a declared start point.

The training range is half-open [train_start, train_stop); the OOS range is [test_start, test_stop). The optional gap excludes the rows immediately before each OOS window from model fitting. The default step equals test_size, so OOS windows do not overlap. A larger step intentionally leaves observations between test windows; smaller steps than test_size are rejected. A partial final window is opt-in and must contain at least two bars.

## Strategy callback contract

The runner accepts two pure-or-controlled callbacks:

- fit(train_bars) returns a fitted model. It receives an immutable tuple containing only the current split's training rows.
- predict(model, history) returns an integer signed-unit target. History is a read-only sequence ending at the current bar and cannot expose later bars through its public Sequence interface.

The target is treated as a signal observed at the current bar close. F39 will not execute it before the next bar open. The callback implementation remains trusted code: the framework constrains what data it receives through its public interface but cannot prove that a callback does not use hidden global state, future-derived labels, or other external leakage. Feature preprocessing and parameter selection must be fitted only within fit() and versioned by the strategy owner.

## Results and metrics

Each OOS window includes its training/test timestamps, split indices, complete F39 BacktestResult, annualized volatility and optional per-regime diagnostics. The summary reports the mean and median window return, population dispersion across window returns, average annualized volatility/drawdown/Sharpe/Sortino, total turnover, trades, commissions and modeled slippage. When regime_by_timestamp is provided, every OOS timestamp must have a non-empty label; labels for unknown timestamps are rejected.

Window returns are summarized, not compounded into one synthetic equity curve: windows can be separated by gaps and each fold starts from initial capital with a flat position. A target signaled on a fold's final bar cannot fill within that fold. This explicit boundary policy avoids carrying a position into data outside the tested interval but can understate strategies that require continuous portfolio state.

A SHA-256 fingerprint covers the input bars, effective split settings, F39 cost settings, declared strategy version, per-window metrics and summary. For a meaningful audit trail, callers should use a stable, changed-on-code-change strategy_version. The fingerprint identifies this evaluation record; it is not a proof of correct strategy implementation.

## Example

    from backtest.walk_forward import WalkForwardConfig, run_walk_forward

    result = run_walk_forward(
        bars,
        WalkForwardConfig(
            initial_train_size=500,
            test_size=100,
            step_size=100,
            mode="rolling",
            gap=0,
        ),
        fit=fit_strategy,
        predict=target_for_history,
        strategy_version="my-strategy-commit-or-version",
    )

    print(result.summary.mean_return)
    print(result.summary.return_dispersion)
    print(result.fingerprint)

The callback names in this example are supplied by the application; they are not bundled model implementations.

## Limitations

- Uses F39's simplified next-bar-open, adverse-slippage and commission assumptions; it does not model partial fills, market impact, funding, borrow costs, corporate actions, exchange calendars or broker constraints.
- A new flat-cash backtest starts at each test fold; the framework does not carry positions or cash across folds.
- Regime tags are inputs, not an inference engine. The caller is responsible for causal, point-in-time regime labels.
- Fit/predict callbacks are user code; leakage through hidden state, non-point-in-time datasets, future-derived labels or preprocessing outside fit() cannot be automatically ruled out.
- OOS averages and dispersion are descriptive diagnostics, not statistical guarantees or investment advice.
- F35–F38 decisions and risk approvals are not automatically wired into the strategy callback; integration must be explicit and independently tested.
