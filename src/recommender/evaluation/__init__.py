"""Candidate generation and evaluation metrics."""

from recommender.evaluation.analysis import (
    BootstrapInterval,
    assign_history_slices,
    bootstrap_confidence_intervals,
    paired_bootstrap_confidence_intervals,
    sliced_means,
)
from recommender.evaluation.candidates import (
    candidate_items,
    generate_candidate_catalog,
)
from recommender.evaluation.metrics import (
    catalog_coverage,
    evaluate_metrics,
    gini_exposure,
    hit_rate_at_k,
    mean_novelty,
    mean_recommended_popularity,
    mrr_at_k,
    ndcg_at_k,
    ranking_metrics,
    recall_at_k,
)

__all__ = [
    "candidate_items",
    "catalog_coverage",
    "evaluate_metrics",
    "gini_exposure",
    "hit_rate_at_k",
    "mean_novelty",
    "mean_recommended_popularity",
    "mrr_at_k",
    "ndcg_at_k",
    "ranking_metrics",
    "recall_at_k",
    "generate_candidate_catalog",
    "BootstrapInterval",
    "assign_history_slices",
    "bootstrap_confidence_intervals",
    "paired_bootstrap_confidence_intervals",
    "sliced_means",
]
