from __future__ import annotations

import pytest

from recommender.evaluation.analysis import (
    assign_history_slices,
    bootstrap_confidence_intervals,
    paired_bootstrap_confidence_intervals,
    sliced_means,
)


def test_bootstrap_intervals_are_reproducible() -> None:
    values = {0: 0.0, 1: 1.0, 2: 1.0}

    first = bootstrap_confidence_intervals(values, replicates=200, seed=7)
    second = bootstrap_confidence_intervals(values, replicates=200, seed=7)

    assert first == second
    assert first.estimate == pytest.approx(2 / 3)
    assert first.low <= first.estimate <= first.high


def test_paired_bootstrap_preserves_user_pairing() -> None:
    leading = {0: 1.0, 1: 0.0, 2: 1.0}
    baseline = {0: 0.0, 1: 0.0, 2: 0.0}

    interval = paired_bootstrap_confidence_intervals(leading, baseline, replicates=200, seed=3)

    assert interval.estimate == pytest.approx(2 / 3)
    assert interval.low <= interval.estimate <= interval.high


def test_history_slices_and_means_use_documented_quantiles() -> None:
    train = [
        {"user_idx": 0, "item_idx": 0},
        {"user_idx": 1, "item_idx": 0},
        {"user_idx": 1, "item_idx": 1},
        {"user_idx": 2, "item_idx": 0},
        {"user_idx": 2, "item_idx": 1},
        {"user_idx": 2, "item_idx": 2},
    ]
    labels = assign_history_slices(train, [0, 1, 2])

    assert set(labels.values()) == {"short", "middle", "long"}
    assert sliced_means({0: 0.0, 1: 0.5, 2: 1.0}, labels) == {
        "short": pytest.approx(0.0),
        "middle": pytest.approx(0.5),
        "long": pytest.approx(1.0),
    }
