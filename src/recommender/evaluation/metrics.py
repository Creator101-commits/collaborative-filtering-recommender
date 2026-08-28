"""Ranking and beyond-accuracy metrics with no sampled negatives."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


def _validate_cutoff(cutoff: int) -> None:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")


def _top(recommended: Sequence[int], cutoff: int) -> Sequence[int]:
    _validate_cutoff(cutoff)
    return recommended[:cutoff]


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], cutoff: int) -> float:
    """Return binary relevance recall for one user."""
    top = _top(recommended, cutoff)
    if not relevant:
        return 0.0
    return len(set(top).intersection(relevant)) / len(set(relevant))


def ndcg_at_k(recommended: Sequence[int], relevant: Collection[int], cutoff: int) -> float:
    """Return binary-relevance NDCG for one user."""
    top = _top(recommended, cutoff)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_idx in enumerate(top)
        if item_idx in relevant_set
    )
    ideal_hits = min(len(relevant_set), cutoff)
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def mrr_at_k(recommended: Sequence[int], relevant: Collection[int], cutoff: int) -> float:
    """Return reciprocal rank of the first relevant item for one user."""
    relevant_set = set(relevant)
    for rank, item_idx in enumerate(_top(recommended, cutoff), start=1):
        if item_idx in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(recommended: Sequence[int], relevant: Collection[int], cutoff: int) -> float:
    """Return one when at least one relevant item is in the top-k list."""
    return float(bool(set(_top(recommended, cutoff)).intersection(relevant)))


def _users(
    recommendations: Mapping[int, Sequence[int]], relevant: Mapping[int, Collection[int]]
) -> tuple[int, ...]:
    return tuple(sorted(set(recommendations).union(relevant)))


def ranking_metrics(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, Collection[int]],
    cutoff: int,
) -> dict[str, float]:
    """Average ranking metrics over the union of evaluated users."""
    users = _users(recommendations, relevant)
    if not users:
        return {
            "recall": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
        }
    values = [
        (
            recall_at_k(recommendations.get(user, ()), relevant.get(user, ()), cutoff),
            ndcg_at_k(recommendations.get(user, ()), relevant.get(user, ()), cutoff),
            mrr_at_k(recommendations.get(user, ()), relevant.get(user, ()), cutoff),
            hit_rate_at_k(recommendations.get(user, ()), relevant.get(user, ()), cutoff),
        )
        for user in users
    ]
    return {
        "recall": sum(value[0] for value in values) / len(values),
        "ndcg": sum(value[1] for value in values) / len(values),
        "mrr": sum(value[2] for value in values) / len(values),
        "hit_rate": sum(value[3] for value in values) / len(values),
    }


def catalog_coverage(
    recommendations: Mapping[int, Sequence[int]], catalog_size: int, cutoff: int
) -> float:
    """Return the fraction of catalog items recommended at least once."""
    _validate_cutoff(cutoff)
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    exposed = {
        item_idx
        for recommendation in recommendations.values()
        for item_idx in recommendation[:cutoff]
    }
    return len(exposed) / catalog_size


def mean_recommended_popularity(
    recommendations: Mapping[int, Sequence[int]],
    item_popularity: Mapping[int, int | float],
    cutoff: int,
) -> float:
    """Return mean training interaction count per recommended exposure."""
    _validate_cutoff(cutoff)
    exposures = [
        item_popularity.get(item_idx, 0.0)
        for recommendation in recommendations.values()
        for item_idx in recommendation[:cutoff]
    ]
    return sum(exposures) / len(exposures) if exposures else 0.0


def mean_novelty(
    recommendations: Mapping[int, Sequence[int]],
    item_popularity: Mapping[int, int | float],
    catalog_size: int,
    cutoff: int,
) -> float:
    """Return mean smoothed self-information of recommended items in bits.

    Laplace smoothing keeps unseen catalog items finite and gives them the
    highest novelty without dropping them from the complete candidate catalog.
    """
    _validate_cutoff(cutoff)
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    total_popularity = sum(max(float(count), 0.0) for count in item_popularity.values())
    denominator = total_popularity + catalog_size
    novelties = []
    for recommendation in recommendations.values():
        for item_idx in recommendation[:cutoff]:
            count = max(float(item_popularity.get(item_idx, 0.0)), 0.0)
            novelties.append(-math.log2((count + 1.0) / denominator))
    return sum(novelties) / len(novelties) if novelties else 0.0


def gini_exposure(
    recommendations: Mapping[int, Sequence[int]], catalog_size: int, cutoff: int
) -> float:
    """Return the Gini coefficient of top-k item exposure."""
    _validate_cutoff(cutoff)
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    exposure = [0] * catalog_size
    for recommendation in recommendations.values():
        for item_idx in recommendation[:cutoff]:
            if not 0 <= item_idx < catalog_size:
                raise ValueError("recommended item is outside the catalog")
            exposure[item_idx] += 1
    total = sum(exposure)
    if total == 0:
        return 0.0
    ordered = sorted(exposure)
    numerator = sum(
        (2 * (index + 1) - catalog_size - 1) * value
        for index, value in enumerate(ordered)
    )
    return numerator / (catalog_size * total)


def evaluate_metrics(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, Collection[int]],
    *,
    catalog_size: int,
    item_popularity: Mapping[int, int | float],
    cutoff: int,
) -> dict[str, float]:
    """Calculate all Phase 4 metrics using the same recommendation lists."""
    ranking = ranking_metrics(recommendations, relevant, cutoff)
    return {
        f"recall@{cutoff}": ranking["recall"],
        f"ndcg@{cutoff}": ranking["ndcg"],
        f"mrr@{cutoff}": ranking["mrr"],
        f"hit_rate@{cutoff}": ranking["hit_rate"],
        f"catalog_coverage@{cutoff}": catalog_coverage(recommendations, catalog_size, cutoff),
        f"mean_popularity@{cutoff}": mean_recommended_popularity(
            recommendations, item_popularity, cutoff
        ),
        f"novelty@{cutoff}": mean_novelty(
            recommendations, item_popularity, catalog_size, cutoff
        ),
        f"gini_exposure@{cutoff}": gini_exposure(recommendations, catalog_size, cutoff),
    }
