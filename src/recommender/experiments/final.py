"""Frozen test evaluation and uncertainty analysis."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from recommender.evaluation import (
    assign_history_slices,
    bootstrap_confidence_intervals,
    evaluate_metrics,
    generate_candidate_catalog,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_confidence_intervals,
    recall_at_k,
    sliced_means,
)
from recommender.experiments.protocol import (
    FINAL_MODEL_NAMES,
    MODEL_HYPERPARAMETER_KEYS,
    PROTOCOL_VERSION,
    ComparisonProtocol,
    ProtocolError,
    read_experiment_card,
)
from recommender.models import (
    ArtifactMetadata,
    BiasedMatrixFactorizationRecommender,
    BPRRecommender,
    ItemKNNRecommender,
    NeighborhoodConfig,
    PopularityRecommender,
    RecommenderProtocol,
    UserKNNRecommender,
)

DEFAULT_FINAL_TRACKING_DIR = Path("mlruns")
DEFAULT_FINAL_ARTIFACT_DIR = Path("artifacts/final")
FINAL_EXPERIMENT_NAME = "movielens-1m-final"


@dataclass(frozen=True, slots=True)
class FinalComparisonResult:
    """Paths produced by the locked final comparison."""

    results_path: Path
    paired_results_path: Path
    slices_path: Path
    tracking_uri: str
    leading_model: str


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = pq.read_table(path).to_pylist()
    except (OSError, pa.ArrowException) as error:
        raise ProtocolError(f"cannot read processed table {path}: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ProtocolError(f"processed table {path} has invalid rows")
    return [cast(dict[str, Any], row) for row in rows]


def _catalog_size(rows: list[dict[str, Any]], field: str, label: str) -> int:
    try:
        values = {int(row[field]) for row in rows}
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(f"{label} has invalid {field} values") from error
    if values != set(range(len(values))):
        raise ProtocolError(f"{label} {field} values are not contiguous")
    return len(values)


def _make_model(
    name: str, user_count: int, item_count: int, config: Mapping[str, object]
) -> RecommenderProtocol:
    unknown = set(config) - MODEL_HYPERPARAMETER_KEYS[name]
    if unknown:
        raise ProtocolError(f"{name} configuration has unknown keys: {sorted(unknown)}")

    def integer(key: str, default: int) -> int:
        value = config.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProtocolError(f"{name} configuration {key} must be an integer")
        return value

    def number(key: str, default: float) -> float:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"{name} configuration {key} must be numeric")
        return float(value)

    if name == PopularityRecommender.model_name:
        return PopularityRecommender(item_count, 4)
    if name in {UserKNNRecommender.model_name, ItemKNNRecommender.model_name}:
        similarity = config.get("similarity", "cosine")
        if not isinstance(similarity, str):
            raise ProtocolError(f"{name} configuration similarity must be a string")
        aggregation = config.get("aggregation", "weighted_mean")
        if not isinstance(aggregation, str):
            raise ProtocolError(f"{name} configuration aggregation must be a string")
        neighborhood = NeighborhoodConfig(
            n_neighbors=integer("n_neighbors", 40),
            similarity=similarity,
            shrinkage=number("shrinkage", 10.0),
            min_support=integer("min_support", 1),
            aggregation=aggregation,
        )
        constructor = (
            UserKNNRecommender if name == UserKNNRecommender.model_name else ItemKNNRecommender
        )
        return constructor(
            user_count,
            item_count,
            n_neighbors=neighborhood.n_neighbors,
            similarity=neighborhood.similarity,
            shrinkage=neighborhood.shrinkage,
            min_support=neighborhood.min_support,
            aggregation=neighborhood.aggregation,
        )
    if name == BiasedMatrixFactorizationRecommender.model_name:
        return BiasedMatrixFactorizationRecommender(
            user_count,
            item_count,
            latent_dim=integer("latent_dim", 32),
            learning_rate=number("learning_rate", 0.01),
            regularization=number("regularization", 0.0001),
            epochs=integer("epochs", 20),
            batch_size=integer("batch_size", 1024),
        )
    if name == BPRRecommender.model_name:
        return BPRRecommender(
            user_count,
            item_count,
            latent_dim=integer("latent_dim", 32),
            learning_rate=number("learning_rate", 0.01),
            regularization=number("regularization", 0.0001),
            epochs=integer("epochs", 20),
            batch_size=integer("batch_size", 1024),
            negative_samples=integer("negative_samples", 1),
            positive_rating_threshold=4,
        )
    raise ProtocolError(f"unsupported final model {name}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"cannot write empty result table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _item_popularity_labels(
    popularity: Mapping[int, int], quantiles: tuple[float, float]
) -> dict[int, str]:
    values = list(popularity.values())
    low, high = (_quantile(values, value) for value in quantiles)
    return {
        item: "tail" if count <= low else "middle" if count <= high else "head"
        for item, count in popularity.items()
    }


def _top_exposure(
    recommendations: Mapping[int, list[int]], popularity: Mapping[int, int], fraction: float
) -> float:
    popular_items = {
        item
        for item, _ in sorted(popularity.items(), key=lambda pair: (-pair[1], pair[0]))[
            : max(1, math.ceil(len(popularity) * fraction))
        ]
    }
    exposures = [item for items in recommendations.values() for item in items]
    return sum(item in popular_items for item in exposures) / len(exposures) if exposures else 0.0


def run_final_comparison(
    processed_dir: Path,
    experiment_card: Path,
    *,
    tracking_dir: Path = DEFAULT_FINAL_TRACKING_DIR,
    artifact_dir: Path = DEFAULT_FINAL_ARTIFACT_DIR,
    protocol: ComparisonProtocol | None = None,
) -> FinalComparisonResult:
    """Evaluate the card's locked models on test and write analysis tables."""
    protocol = protocol or ComparisonProtocol()
    card = read_experiment_card(experiment_card, protocol=protocol)
    processed_dir = processed_dir.resolve()
    manifest_path = processed_dir / "manifest.json"
    try:
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ProtocolError(f"cannot read processed manifest {manifest_path}: {error}") from error
    if card["data_manifest_sha256"] != manifest_hash:
        raise ProtocolError("experiment card does not match the processed manifest")

    train_rows = _read_rows(processed_dir / "train_for_test.parquet")
    test_rows = _read_rows(processed_dir / "test.parquet")
    item_count = _catalog_size(_read_rows(processed_dir / "movies.parquet"), "item_idx", "movies")
    user_count = _catalog_size(_read_rows(processed_dir / "users.parquet"), "user_idx", "users")
    relevant: dict[int, set[int]] = {}
    for source_row in test_rows:
        user = int(source_row["user_idx"])
        item = int(source_row["item_idx"])
        if int(source_row["rating"]) < 4:
            raise ProtocolError("test contains a non-relevant held-out interaction")
        relevant.setdefault(user, set()).add(item)
    candidates = generate_candidate_catalog(item_count, train_rows, user_indices=sorted(relevant))
    if any(not candidates[user] for user in relevant):
        raise ProtocolError("a test user has no eligible candidates")
    popularity = Counter(int(row["item_idx"]) for row in train_rows if int(row["rating"]) >= 4)
    item_popularity = {item: popularity.get(item, 0) for item in range(item_count)}
    item_labels = _item_popularity_labels(item_popularity, protocol.item_popularity_quantiles)

    tracking_dir = tracking_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(FINAL_EXPERIMENT_NAME)

    model_configs = cast(dict[str, object], card["models"])
    all_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    per_model_metrics: dict[str, dict[str, dict[int, float]]] = {}
    history_labels = assign_history_slices(
        train_rows, relevant, quantiles=protocol.history_quantiles
    )
    learned_models = {"matrix_factorization", "bpr"}

    for model_name in FINAL_MODEL_NAMES:
        config = cast(Mapping[str, object], model_configs[model_name])
        seeds = protocol.learned_model_seeds if model_name in learned_models else (0,)
        per_model_metrics[model_name] = {
            metric: {} for metric in ("recall", "ndcg", "mrr", "hit_rate")
        }
        for seed_number, seed in enumerate(seeds):
            model = _make_model(model_name, user_count, item_count, config)
            train_started = time.perf_counter()
            model.fit(train_rows, seed=seed)
            train_seconds = time.perf_counter() - train_started
            inference_started = time.perf_counter()
            recommendations = {
                user: list(model.recommend(user, candidates[user], max(protocol.metric_cutoffs)))
                for user in sorted(relevant)
            }
            inference_seconds = time.perf_counter() - inference_started
            user_metrics = {
                user: {
                    "recall": recall_at_k(
                        recommendations[user], relevant[user], protocol.primary_cutoff
                    ),
                    "ndcg": ndcg_at_k(
                        recommendations[user], relevant[user], protocol.primary_cutoff
                    ),
                    "mrr": mrr_at_k(recommendations[user], relevant[user], protocol.primary_cutoff),
                    "hit_rate": hit_rate_at_k(
                        recommendations[user], relevant[user], protocol.primary_cutoff
                    ),
                }
                for user in relevant
            }
            for metric in per_model_metrics[model_name]:
                for user, values in user_metrics.items():
                    previous = per_model_metrics[model_name][metric].get(user, 0.0)
                    per_model_metrics[model_name][metric][user] = (
                        previous * seed_number + values[metric]
                    ) / (seed_number + 1)

            with mlflow.start_run(run_name=f"{model_name}-test-seed-{seed}") as active_run:
                run_id = active_run.info.run_id
                metadata = ArtifactMetadata(
                    model_name=model_name,
                    data_manifest_sha256=manifest_hash,
                    configuration={
                        **dict(config),
                        "protocol_version": PROTOCOL_VERSION,
                        "split": "test",
                    },
                    seed=seed,
                    git_revision=cast(str, card["git_revision"]),
                    run_id=run_id,
                )
                suffix = ".pt" if model_name in learned_models else ".json"
                model_path = artifact_dir / f"{run_id}-{model_name}-test-seed-{seed}{suffix}"
                model.save(model_path, metadata=metadata)
                result_row: dict[str, object] = {
                    "artifact_path": str(model_path),
                    "artifact_size_bytes": model_path.stat().st_size,
                    "data_manifest_sha256": manifest_hash,
                    "git_revision": card["git_revision"],
                    "inference_seconds": inference_seconds,
                    "model": model_name,
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": run_id,
                    "seed": seed,
                    "split": "test",
                    "train_seconds": train_seconds,
                    "users": len(relevant),
                }
                for cutoff in protocol.metric_cutoffs:
                    result_row.update(
                        evaluate_metrics(
                            recommendations,
                            relevant,
                            catalog_size=item_count,
                            item_popularity=item_popularity,
                            cutoff=cutoff,
                        )
                    )
                for metric in ("recall", "ndcg", "mrr", "hit_rate"):
                    interval = bootstrap_confidence_intervals(
                        {user: values[metric] for user, values in user_metrics.items()},
                        replicates=protocol.bootstrap_replicates,
                        confidence_level=protocol.confidence_level,
                        seed=seed,
                    )
                    prefix = f"{metric}@{protocol.primary_cutoff}"
                    result_row[f"{prefix}_estimate"] = interval.estimate
                    result_row[f"{prefix}_ci_low"] = interval.low
                    result_row[f"{prefix}_ci_high"] = interval.high
                mlflow.set_tags(
                    {
                        "data_manifest_sha256": manifest_hash,
                        "git_revision": cast(str, card["git_revision"]),
                        "model": model_name,
                        "protocol_version": PROTOCOL_VERSION,
                        "split": "test",
                    }
                )
                mlflow.log_params({"seed": seed, "protocol_version": PROTOCOL_VERSION})
                mlflow.log_metrics(
                    {
                        key.replace("@", "_at_"): float(value)
                        for key, value in result_row.items()
                        if "@" in key and isinstance(value, (int, float))
                    }
                )
                mlflow.log_artifact(str(model_path), artifact_path="model")
            all_rows.append(result_row)

            for metric in ("recall", "ndcg", "mrr", "hit_rate"):
                means = {user: values[metric] for user, values in user_metrics.items()}
                for label, value in sliced_means(means, history_labels).items():
                    slice_rows.append(
                        {
                            "estimate": value,
                            "label": label,
                            "metric": metric,
                            "model": model_name,
                            "seed": seed,
                            "slice": "history_length",
                        }
                    )
            exposures = Counter(
                item
                for items in recommendations.values()
                for item in items[: protocol.primary_cutoff]
            )
            total_exposures = sum(exposures.values())
            for label in ("tail", "middle", "head"):
                exposure = sum(
                    count for item, count in exposures.items() if item_labels[item] == label
                )
                slice_rows.append(
                    {
                        "estimate": exposure / total_exposures if total_exposures else 0.0,
                        "label": label,
                        "metric": "exposure",
                        "model": model_name,
                        "seed": seed,
                        "slice": "item_popularity",
                    }
                )
            for fraction in (0.01, 0.05, 0.10):
                slice_rows.append(
                    {
                        "estimate": _top_exposure(recommendations, item_popularity, fraction),
                        "label": f"top_{fraction:.0%}",
                        "metric": "exposure",
                        "model": model_name,
                        "seed": seed,
                        "slice": "recommendation_concentration",
                    }
                )

    leading_model = max(
        FINAL_MODEL_NAMES,
        key=lambda name: (
            sum(per_model_metrics[name]["ndcg"].values()) / len(per_model_metrics[name]["ndcg"])
        ),
    )
    paired_rows: list[dict[str, object]] = []
    for baseline in FINAL_MODEL_NAMES:
        if baseline == leading_model:
            continue
        for metric in ("recall", "ndcg", "mrr", "hit_rate"):
            interval = paired_bootstrap_confidence_intervals(
                per_model_metrics[leading_model][metric],
                per_model_metrics[baseline][metric],
                replicates=protocol.bootstrap_replicates,
                confidence_level=protocol.confidence_level,
                seed=0,
            )
            paired_rows.append(
                {
                    "baseline_model": baseline,
                    "ci_high": interval.high,
                    "ci_low": interval.low,
                    "estimate": interval.estimate,
                    "leading_model": leading_model,
                    "metric": f"{metric}@{protocol.primary_cutoff}",
                }
            )

    results_path = artifact_dir / "final-results.csv"
    paired_path = artifact_dir / "paired-results.csv"
    slices_path = artifact_dir / "final-slices.csv"
    _write_csv(results_path, all_rows)
    _write_csv(paired_path, paired_rows)
    _write_csv(slices_path, slice_rows)
    return FinalComparisonResult(
        results_path=results_path,
        paired_results_path=paired_path,
        slices_path=slices_path,
        tracking_uri=tracking_dir.as_uri(),
        leading_model=leading_model,
    )
