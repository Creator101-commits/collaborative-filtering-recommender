"""User-level uncertainty and slice helpers for final evaluation."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A percentile interval around a user-level mean."""

    estimate: float
    low: float
    high: float


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _validate_bootstrap(replicates: int, confidence_level: float) -> None:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")


def bootstrap_confidence_intervals(
    values: Mapping[int, float],
    *,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a reproducible percentile bootstrap interval for a user mean."""
    _validate_bootstrap(replicates, confidence_level)
    if not values:
        raise ValueError("values must not be empty")
    user_ids = tuple(sorted(values))
    estimate = sum(float(values[user]) for user in user_ids) / len(user_ids)
    generator = random.Random(seed)
    bootstrap_means = [
        sum(float(values[user]) for user in generator.choices(user_ids, k=len(user_ids)))
        / len(user_ids)
        for _ in range(replicates)
    ]
    alpha = (1 - confidence_level) / 2
    return BootstrapInterval(
        estimate=estimate,
        low=_percentile(bootstrap_means, alpha),
        high=_percentile(bootstrap_means, 1 - alpha),
    )


def paired_bootstrap_confidence_intervals(
    leading: Mapping[int, float],
    baseline: Mapping[int, float],
    *,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a paired percentile interval for leading-minus-baseline means."""
    if set(leading) != set(baseline):
        raise ValueError("leading and baseline must contain the same users")
    differences = {user: leading[user] - baseline[user] for user in leading}
    return bootstrap_confidence_intervals(
        differences,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )


def _quantile(values: list[int], quantile: float) -> float:
    return _percentile([float(value) for value in values], quantile)


def assign_history_slices(
    training_rows: Iterable[Mapping[str, object]],
    user_indices: Iterable[int],
    *,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
) -> dict[int, str]:
    """Assign short, middle, and long labels using training-count quantiles."""
    if not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("quantiles must be increasing and between 0 and 1")
    counts = Counter(int(cast(Any, row["user_idx"])) for row in training_rows)
    users = tuple(sorted(set(user_indices)))
    if not users:
        return {}
    values = [counts[user] for user in users]
    low, high = (_quantile(values, quantile) for quantile in quantiles)
    return {
        user: "short" if counts[user] <= low else "middle" if counts[user] <= high else "long"
        for user in users
    }


def sliced_means(values: Mapping[int, float], labels: Mapping[int, str]) -> dict[str, float]:
    """Average user values within each supplied slice label."""
    grouped: dict[str, list[float]] = {}
    for user, label in labels.items():
        if user in values:
            grouped.setdefault(label, []).append(float(values[user]))
    return {
        label: sum(group_values) / len(group_values)
        for label, group_values in sorted(grouped.items())
        if group_values
    }
