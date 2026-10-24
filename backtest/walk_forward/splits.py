"""Deterministic temporal split generation for walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WalkForwardMode = Literal["rolling", "expanding", "anchored"]


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Temporal split policy.

    rolling keeps a fixed-size training window; expanding uses the full prefix;
    anchored expands from anchor_index. Test windows never overlap.
    """

    initial_train_size: int
    test_size: int
    step_size: int | None = None
    mode: WalkForwardMode = "expanding"
    gap: int = 0
    anchor_index: int = 0
    allow_partial_test: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_train_size", self.initial_train_size),
            ("test_size", self.test_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer greater than or equal to 2")
        if self.step_size is not None:
            if (
                not isinstance(self.step_size, int)
                or isinstance(self.step_size, bool)
                or self.step_size < self.test_size
            ):
                raise ValueError("step_size must be an integer greater than or equal to test_size")
        if self.mode not in ("rolling", "expanding", "anchored"):
            raise ValueError("mode must be rolling, expanding or anchored")
        if not isinstance(self.gap, int) or isinstance(self.gap, bool) or self.gap < 0:
            raise ValueError("gap must be a non-negative integer")
        if (
            not isinstance(self.anchor_index, int)
            or isinstance(self.anchor_index, bool)
            or self.anchor_index < 0
        ):
            raise ValueError("anchor_index must be a non-negative integer")
        if not isinstance(self.allow_partial_test, bool):
            raise ValueError("allow_partial_test must be boolean")

    @property
    def effective_step_size(self) -> int:
        """Return the configured step, defaulting to a disjoint test window."""
        return self.test_size if self.step_size is None else self.step_size


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Half-open, positional ranges; stop indices are excluded."""

    train_start: int
    train_stop: int
    test_start: int
    test_stop: int

    @property
    def train_size(self) -> int:
        return self.train_stop - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_stop - self.test_start


def make_splits(n_samples: int, config: WalkForwardConfig) -> tuple[TemporalSplit, ...]:
    """Build chronological splits without overlapping OOS windows or using future rows."""
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer")

    anchor = config.anchor_index if config.mode == "anchored" else 0
    first_test = anchor + config.initial_train_size + config.gap
    if first_test >= n_samples:
        raise ValueError("not enough samples for the initial training window and gap")

    splits: list[TemporalSplit] = []
    test_start = first_test
    while test_start < n_samples:
        remaining = n_samples - test_start
        if remaining >= config.test_size:
            test_stop = test_start + config.test_size
        elif config.allow_partial_test and remaining >= 2:
            test_stop = n_samples
        else:
            break

        train_stop = test_start - config.gap
        if config.mode == "rolling":
            train_start = train_stop - config.initial_train_size
        elif config.mode == "anchored":
            train_start = config.anchor_index
        else:
            train_start = 0

        if train_start < 0 or train_stop - train_start < config.initial_train_size:
            raise ValueError("split cannot satisfy the minimum training size")
        if train_stop > test_start:
            raise AssertionError("internal error: training data overlaps the test window")

        splits.append(
            TemporalSplit(
                train_start=train_start,
                train_stop=train_stop,
                test_start=test_start,
                test_stop=test_stop,
            )
        )
        test_start += config.effective_step_size

    if not splits:
        raise ValueError("not enough samples to create an out-of-sample test window")
    return tuple(splits)
