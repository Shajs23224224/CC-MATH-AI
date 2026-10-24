from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from backtest.engine import BacktestConfig, Bar
from backtest.walk_forward import WalkForwardConfig, make_splits, run_walk_forward


def bars(*closes: float) -> list[Bar]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        Bar(start + timedelta(days=index), close, close, close, close, 1000.0)
        for index, close in enumerate(closes)
    ]


def test_rolling_uses_fixed_length_training_windows_and_respects_gap() -> None:
    config = WalkForwardConfig(
        initial_train_size=3,
        test_size=2,
        step_size=2,
        mode="rolling",
        gap=1,
    )
    splits = make_splits(10, config)
    assert [(part.train_start, part.train_stop) for part in splits] == [(0, 3), (2, 5), (4, 7)]
    assert [(part.test_start, part.test_stop) for part in splits] == [(4, 6), (6, 8), (8, 10)]
    assert all(part.train_stop <= part.test_start for part in splits)


def test_expanding_windows_grow_training_prefix_and_oos_windows_do_not_overlap() -> None:
    splits = make_splits(
        10,
        WalkForwardConfig(initial_train_size=3, test_size=2, step_size=2, mode="expanding"),
    )
    assert [(part.train_start, part.train_stop) for part in splits] == [(0, 3), (0, 5), (0, 7)]
    assert [(part.test_start, part.test_stop) for part in splits] == [(3, 5), (5, 7), (7, 9)]


def test_anchored_training_starts_at_explicit_anchor() -> None:
    splits = make_splits(
        12,
        WalkForwardConfig(
            initial_train_size=3,
            test_size=2,
            step_size=2,
            mode="anchored",
            anchor_index=2,
            gap=1,
        ),
    )
    assert splits[0].train_start == 2
    assert splits[0].train_stop == 5
    assert splits[0].test_start == 6
    assert splits[1].train_start == 2
    assert splits[1].train_stop == 7


def test_partial_final_window_is_opt_in_and_must_have_at_least_two_rows() -> None:
    config = WalkForwardConfig(
        initial_train_size=3,
        test_size=3,
        step_size=3,
        allow_partial_test=True,
    )
    splits = make_splits(8, config)
    assert [(part.test_start, part.test_stop) for part in splits] == [(3, 6), (6, 8)]


def test_config_rejects_overlapping_test_windows_and_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="step_size"):
        WalkForwardConfig(initial_train_size=3, test_size=3, step_size=2)
    with pytest.raises(ValueError, match="test_size"):
        WalkForwardConfig(initial_train_size=3, test_size=1)
    with pytest.raises(ValueError, match="gap"):
        WalkForwardConfig(initial_train_size=3, test_size=2, gap=-1)


def test_split_generation_rejects_insufficient_data() -> None:
    config = WalkForwardConfig(initial_train_size=4, test_size=2)
    with pytest.raises(ValueError, match="not enough samples"):
        make_splits(5, config)


def test_fit_receives_training_only_and_predict_sees_causal_history() -> None:
    series = bars(100, 101, 102, 103, 104, 105, 106, 107, 108)
    fitted_on: list[tuple[datetime, ...]] = []
    predicted_at: list[datetime] = []

    def fit(train: tuple[Bar, ...]) -> datetime:
        fitted_on.append(tuple(bar.timestamp for bar in train))
        return train[-1].timestamp

    def predict(model: datetime, history: Sequence[Bar]) -> int:
        # Use the Sequence interface without access to the underlying full series.
        observations = tuple(history)
        current = observations[-1]
        assert current.timestamp > model
        assert all(
            left.timestamp < right.timestamp
            for left, right in zip(observations, observations[1:], strict=False)
        )
        predicted_at.append(current.timestamp)
        try:
            history[len(observations)]
        except IndexError:
            pass
        else:
            raise AssertionError("future row leaked through history view")
        return 1

    result = run_walk_forward(
        series,
        WalkForwardConfig(initial_train_size=3, test_size=2, step_size=2, mode="expanding"),
        fit=fit,
        predict=predict,
        backtest_config=BacktestConfig(commission_bps=0, slippage_bps=0),
        strategy_version="test-strategy-v1",
    )

    assert [len(window) for window in fitted_on] == [3, 5, 7]
    assert len(predicted_at) == 6
    assert result.summary.window_count == 3
    assert all(window.backtest.trades == 1 for window in result.windows)
    assert len(result.fingerprint) == 64


def test_predict_must_return_signed_integer_units_not_boolean_or_float() -> None:
    series = bars(100, 101, 102, 103, 104)
    config = WalkForwardConfig(initial_train_size=3, test_size=2)
    with pytest.raises(ValueError, match="integer signed-unit target"):
        run_walk_forward(series, config, fit=lambda train: 1, predict=lambda model, history: 1.5)


def test_regime_metrics_are_computed_per_label_and_require_complete_oos_coverage() -> None:
    series = bars(100, 101, 102, 103, 104, 105, 106)
    regimes = {
        bar.timestamp: ("bull" if index >= 5 else "bear") for index, bar in enumerate(series)
    }
    result = run_walk_forward(
        series,
        WalkForwardConfig(initial_train_size=3, test_size=2),
        fit=lambda train: len(train),
        predict=lambda model, history: 1,
        regime_by_timestamp=regimes,
        strategy_version="regime-test-v1",
    )
    labels = {metric.label for metric in result.summary.regime_metrics}
    assert labels == {"bear", "bull"}
    assert all(metric.observations > 0 for metric in result.summary.regime_metrics)

    incomplete = {
        timestamp: label for timestamp, label in regimes.items() if timestamp != series[3].timestamp
    }
    with pytest.raises(ValueError, match="every out-of-sample timestamp"):
        run_walk_forward(
            series,
            WalkForwardConfig(initial_train_size=3, test_size=2),
            fit=lambda train: len(train),
            predict=lambda model, history: 0,
            regime_by_timestamp=incomplete,
        )


def test_regime_metadata_rejects_unknown_timestamps_and_empty_labels() -> None:
    series = bars(100, 101, 102, 103, 104)
    unknown = datetime(2030, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="timestamps present in bars"):
        run_walk_forward(
            series,
            WalkForwardConfig(initial_train_size=3, test_size=2),
            fit=lambda train: len(train),
            predict=lambda model, history: 0,
            regime_by_timestamp={unknown: "bull"},
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        run_walk_forward(
            series,
            WalkForwardConfig(initial_train_size=3, test_size=2),
            fit=lambda train: len(train),
            predict=lambda model, history: 0,
            regime_by_timestamp={series[3].timestamp: " "},
        )


def test_fingerprint_is_repeatable_for_identical_inputs_and_version() -> None:
    series = bars(100, 101, 99, 102, 104, 103, 105, 108)

    def run() -> str:
        return run_walk_forward(
            series,
            WalkForwardConfig(initial_train_size=3, test_size=2),
            fit=lambda train: len(train),
            predict=lambda model, history: 1 if history[-1].close >= 100 else 0,
            strategy_version="deterministic-v1",
        ).fingerprint

    assert run() == run()


def test_strategy_version_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="strategy_version"):
        run_walk_forward(
            bars(100, 101, 102, 103, 104),
            WalkForwardConfig(initial_train_size=3, test_size=2),
            fit=lambda train: len(train),
            predict=lambda model, history: 0,
            strategy_version=" ",
        )
