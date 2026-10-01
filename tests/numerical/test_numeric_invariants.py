import math


def test_simple_return_reference_value() -> None:
    initial = 100.0
    final = 110.0
    simple_return = (final / initial) - 1.0
    assert math.isclose(simple_return, 0.10, rel_tol=0.0, abs_tol=1e-12)


def test_log_return_reference_value() -> None:
    initial = 100.0
    final = 110.0
    log_return = math.log(final / initial)
    assert math.isclose(log_return, math.log(1.10), rel_tol=0.0, abs_tol=1e-12)


def test_simple_and_log_return_have_consistent_direction() -> None:
    for initial, final in [(80.0, 100.0), (100.0, 80.0), (100.0, 100.0)]:
        simple_return = (final / initial) - 1.0
        log_return = math.log(final / initial)
        if final == initial:
            assert simple_return == 0.0
            assert log_return == 0.0
        else:
            assert (simple_return > 0) == (log_return > 0)


def test_portfolio_weight_sum_can_be_bounded() -> None:
    weights = (0.25, 0.35, 0.40)
    assert all(0.0 <= weight <= 1.0 for weight in weights)
    assert math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
