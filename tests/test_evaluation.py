"""Tests for full-catalog candidates and Phase 4 metrics."""

from __future__ import annotations

import math

import pytest

from recommender.evaluation import (
    catalog_coverage,
    evaluate_metrics,
    generate_candidate_catalog,
    gini_exposure,
    hit_rate_at_k,
    mean_novelty,
    mean_recommended_popularity,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_candidates_remove_only_observed_training_items() -> None:
    train = [
        {"user_idx": 0, "item_idx": 0},
        {"user_idx": 0, "item_idx": 3},
    ]

    candidates = generate_candidate_catalog(5, train, user_indices=[0, 1])

    assert candidates[0] == (1, 2, 4)
    assert candidates[1] == (0, 1, 2, 3, 4)


def test_candidates_reject_out_of_catalog_items() -> None:
    with pytest.raises(ValueError, match="outside the catalog"):
        generate_candidate_catalog(2, [{"user_idx": 0, "item_idx": 2}])


def test_ranking_metrics_use_hand_calculable_binary_relevance() -> None:
    recommended = [2, 1, 3]
    relevant = {1}

    assert recall_at_k(recommended, relevant, 2) == 1.0
    assert ndcg_at_k(recommended, relevant, 2) == pytest.approx(1 / math.log2(3))
    assert mrr_at_k(recommended, relevant, 2) == 0.5
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0


def test_beyond_accuracy_metrics_use_top_k_exposure() -> None:
    recommendations = {0: [2, 1, 3], 1: [0, 4]}
    popularity = {0: 4, 1: 3, 2: 2, 3: 0, 4: 1}

    assert catalog_coverage(recommendations, 5, 2) == pytest.approx(4 / 5)
    assert mean_recommended_popularity(recommendations, popularity, 2) == pytest.approx(2.5)
    assert gini_exposure({0: [0], 1: [0]}, 5, 1) == pytest.approx(0.8)

    novelty = mean_novelty(recommendations, popularity, 5, 2)
    assert novelty > 0
    assert novelty == pytest.approx(
        sum(-math.log2((popularity[item] + 1) / 15) for item in (2, 1, 0, 4)) / 4
    )


def test_aggregate_metrics_cover_all_users() -> None:
    metrics = evaluate_metrics(
        {0: [1], 1: [2]},
        {0: {1}, 1: {3}},
        catalog_size=4,
        item_popularity={0: 0, 1: 2, 2: 1, 3: 0},
        cutoff=1,
    )

    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["ndcg@1"] == pytest.approx(0.5)
    assert metrics["mrr@1"] == pytest.approx(0.5)
    assert metrics["hit_rate@1"] == pytest.approx(0.5)
    assert metrics["catalog_coverage@1"] == pytest.approx(0.5)
