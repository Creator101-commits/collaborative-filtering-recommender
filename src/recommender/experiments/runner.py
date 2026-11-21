"""Validation-only experiment runners for reproducible recommender baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from recommender.evaluation import evaluate_metrics, generate_candidate_catalog
from recommender.experiments.protocol import PROTOCOL_VERSION
from recommender.models import (
    ArtifactMetadata,
    BiasedMatrixFactorizationRecommender,
    BPRConfig,
    BPRRecommender,
    ItemKNNRecommender,
    MatrixFactorizationConfig,
    NeighborhoodConfig,
    PopularityRecommender,
    RecommenderProtocol,
    UserKNNRecommender,
    ValidationAwareRecommenderProtocol,
)

DEFAULT_TRACKING_DIR = Path("mlruns")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_EXPERIMENT_NAME = "movielens-1m-baseline"


class ExperimentError(RuntimeError):
    """Raised when a reproducible experiment cannot be run."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Settings frozen into a validation run and its model artifact."""

    cutoff: int = 10
    positive_rating_threshold: int = 4
    seed: int = 0
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if not 1 <= self.positive_rating_threshold <= 5:
            raise ValueError("positive_rating_threshold must be between 1 and 5")
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")

    def as_dict(self) -> dict[str, object]:
        return {
            "cutoff": self.cutoff,
            "experiment_name": self.experiment_name,
            "protocol_version": self.protocol_version,
            "positive_rating_threshold": self.positive_rating_threshold,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Paths and metrics produced by one validation run."""

    run_id: str
    metrics: dict[str, float]
    model_path: Path
    results_path: Path
    tracking_uri: str
    data_manifest_sha256: str
    git_revision: str
    train_seconds: float
    inference_seconds: float
    best_epoch: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentError(f"cannot read processed manifest {path}: {error}") from error
    if not isinstance(value, dict) or value.get("dataset") != "MovieLens 1M":
        raise ExperimentError("processed manifest is missing or unsupported")
    return cast(dict[str, object], value)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = pq.read_table(path).to_pylist()
    except (OSError, pa.ArrowException) as error:
        raise ExperimentError(f"cannot read processed table {path}: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ExperimentError(f"processed table {path} has invalid rows")
    return [cast(dict[str, Any], row) for row in rows]


def _git_revision() -> str:
    repository = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "unknown"


def _indexed_count(rows: list[dict[str, Any]], field: str, label: str) -> int:
    indices: list[int] = []
    for row in rows:
        try:
            indices.append(int(row[field]))
        except (KeyError, TypeError, ValueError) as error:
            raise ExperimentError(f"{label} table has an invalid {field}") from error
    if not indices or set(indices) != set(range(len(indices))):
        raise ExperimentError(f"{label} table {field} values are not a contiguous catalog")
    return len(indices)


def _relevant_by_user(rows: list[dict[str, Any]]) -> dict[int, set[int]]:
    relevant: defaultdict[int, set[int]] = defaultdict(set)
    for row in rows:
        try:
            user_idx = int(row["user_idx"])
            item_idx = int(row["item_idx"])
        except (KeyError, TypeError, ValueError) as error:
            raise ExperimentError("validation table has invalid user or item indices") from error
        relevant[user_idx].add(item_idx)
    return dict(relevant)


def _training_popularity(
    rows: list[dict[str, Any]], item_count: int, positive_rating_threshold: int
) -> dict[int, int]:
    counts = [0] * item_count
    for row in rows:
        try:
            item_idx = int(row["item_idx"])
            rating = int(row["rating"])
        except (KeyError, TypeError, ValueError) as error:
            raise ExperimentError("training table has invalid item or rating values") from error
        if not 0 <= item_idx < item_count:
            raise ExperimentError("training table contains an item outside the catalog")
        if rating >= positive_rating_threshold:
            counts[item_idx] += 1
    return dict(enumerate(counts))


def _mlflow_metric_key(name: str) -> str:
    return name.replace("@", "_at_")


def _write_result_table(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(dict(row))


def _run_validation_model(
    processed_dir: Path,
    *,
    model_name: str,
    model_factory: Callable[[int, int], RecommenderProtocol],
    model_configuration: Mapping[str, object],
    run_name: str,
    tracking_dir: Path,
    artifact_dir: Path,
    config: ExperimentConfig,
    checkpoint_on_validation: bool = False,
) -> ExperimentResult:
    """Fit one model on train and evaluate it against validation only."""
    processed_dir = processed_dir.resolve()
    manifest_path = processed_dir / "manifest.json"
    _read_manifest(manifest_path)
    manifest_hash = _sha256(manifest_path)
    train_rows = _read_rows(processed_dir / "train.parquet")
    validation_rows = _read_rows(processed_dir / "validation.parquet")
    item_count = _indexed_count(_read_rows(processed_dir / "movies.parquet"), "item_idx", "movies")
    user_count = _indexed_count(_read_rows(processed_dir / "users.parquet"), "user_idx", "users")
    relevant = _relevant_by_user(validation_rows)
    if not relevant:
        raise ExperimentError("validation table has no evaluated users")
    if any(user_idx < 0 or user_idx >= user_count for user_idx in relevant):
        raise ExperimentError("validation table contains a user outside the catalog")
    candidates = generate_candidate_catalog(
        item_count,
        train_rows,
        user_indices=relevant,
    )
    if any(not candidates[user_idx] for user_idx in relevant):
        raise ExperimentError("an evaluated user has no eligible candidates")

    model = model_factory(user_count, item_count)
    train_started = time.perf_counter()
    if checkpoint_on_validation:
        validation_model = cast(ValidationAwareRecommenderProtocol, model)
        validation_model.fit(
            train_rows, seed=config.seed, validation_interactions=validation_rows
        )
    else:
        model.fit(train_rows, seed=config.seed)
    train_seconds = time.perf_counter() - train_started
    best_epoch = getattr(model, "best_epoch", None)
    if best_epoch is not None and not isinstance(best_epoch, int):
        raise ExperimentError(f"{model_name} best_epoch is not an integer")

    inference_started = time.perf_counter()
    recommendations = {
        user_idx: model.recommend(user_idx, candidates[user_idx], config.cutoff)
        for user_idx in sorted(relevant)
    }
    inference_seconds = time.perf_counter() - inference_started
    metrics = evaluate_metrics(
        recommendations,
        relevant,
        catalog_size=item_count,
        item_popularity=_training_popularity(
            train_rows, item_count, config.positive_rating_threshold
        ),
        cutoff=config.cutoff,
    )

    tracking_dir = tracking_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_dir.as_uri()
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    git_revision = _git_revision()
    configuration = {
        **config.as_dict(),
        "model": model_name,
        **dict(model_configuration),
    }

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id
        metadata = ArtifactMetadata(
            model_name=model_name,
            data_manifest_sha256=manifest_hash,
            configuration=configuration,
            seed=config.seed,
            git_revision=git_revision,
            run_id=run_id,
        )
        suffix = (
            ".pt"
            if model_name in {
                BiasedMatrixFactorizationRecommender.model_name,
                BPRRecommender.model_name,
            }
            else ".json"
        )
        model_path = artifact_dir / f"{run_id}-{model_name}{suffix}"
        model.save(model_path, metadata=metadata)
        result_row: dict[str, object] = {
            "catalog_size": item_count,
            "cutoff": config.cutoff,
            "data_manifest_sha256": manifest_hash,
            "git_revision": git_revision,
            "inference_seconds": inference_seconds,
            **dict(model_configuration),
            "model": model_name,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "seed": config.seed,
            "split": "validation",
            "train_seconds": train_seconds,
            "users": len(relevant),
            **metrics,
            "positive_rating_threshold": config.positive_rating_threshold,
        }
        results_path = artifact_dir / f"{run_id}-results.csv"
        _write_result_table(results_path, result_row)
        mlflow.log_params(
            {
                **config.as_dict(),
                "data_manifest_sha256": manifest_hash,
                "git_revision": git_revision,
                "model": model_name,
                **dict(model_configuration),
                "split": "validation",
            }
        )
        mlflow.log_metrics(
            {
                **{_mlflow_metric_key(name): value for name, value in metrics.items()},
                "inference_seconds": inference_seconds,
                "train_seconds": train_seconds,
            }
        )
        mlflow.set_tags(
            {
                "data_manifest_sha256": manifest_hash,
                "protocol_version": PROTOCOL_VERSION,
                "git_revision": git_revision,
                "model": model_name,
                "split": "validation",
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(results_path), artifact_path="results")
        mlflow.log_artifact(str(manifest_path), artifact_path="data")

    return ExperimentResult(
        run_id=run_id,
        metrics=metrics,
        model_path=model_path,
        results_path=results_path,
        tracking_uri=tracking_uri,
        data_manifest_sha256=manifest_hash,
        git_revision=git_revision,
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        best_epoch=best_epoch,
    )


def run_popularity_baseline(
    processed_dir: Path,
    *,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run the popularity baseline through the common validation runner."""
    config = config or ExperimentConfig()
    return _run_validation_model(
        processed_dir,
        model_name=PopularityRecommender.model_name,
        model_factory=lambda _user_count, item_count: PopularityRecommender(
            item_count, config.positive_rating_threshold
        ),
        model_configuration={},
        run_name=f"popularity-validation-seed-{config.seed}",
        tracking_dir=tracking_dir,
        artifact_dir=artifact_dir,
        config=config,
    )


def run_neighborhood_baseline(
    processed_dir: Path,
    *,
    model: str = "user_knn",
    n_neighbors: int = 40,
    similarity: str = "cosine",
    shrinkage: float = 10.0,
    min_support: int = 1,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run one user- or item-KNN model through the common validation runner."""
    if model not in {UserKNNRecommender.model_name, ItemKNNRecommender.model_name}:
        raise ValueError("model must be user_knn or item_knn")
    neighborhood = NeighborhoodConfig(
        n_neighbors=n_neighbors,
        similarity=similarity,
        shrinkage=shrinkage,
        min_support=min_support,
    )
    config = config or ExperimentConfig()
    def make_user_model(user_count: int, item_count: int) -> RecommenderProtocol:
        return UserKNNRecommender(
            user_count,
            item_count,
            n_neighbors=neighborhood.n_neighbors,
            similarity=neighborhood.similarity,
            shrinkage=neighborhood.shrinkage,
            min_support=neighborhood.min_support,
        )

    def make_item_model(user_count: int, item_count: int) -> RecommenderProtocol:
        return ItemKNNRecommender(
            user_count,
            item_count,
            n_neighbors=neighborhood.n_neighbors,
            similarity=neighborhood.similarity,
            shrinkage=neighborhood.shrinkage,
            min_support=neighborhood.min_support,
        )

    factory = make_user_model if model == UserKNNRecommender.model_name else make_item_model
    return _run_validation_model(
        processed_dir,
        model_name=model,
        model_factory=factory,
        model_configuration=neighborhood.as_dict(),
        run_name=f"{model}-validation-seed-{config.seed}",
        tracking_dir=tracking_dir,
        artifact_dir=artifact_dir,
        config=config,
    )


def run_matrix_factorization_baseline(
    processed_dir: Path,
    *,
    latent_dim: int = 32,
    learning_rate: float = 0.01,
    regularization: float = 0.0001,
    epochs: int = 20,
    batch_size: int = 1024,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run biased matrix factorization through the common validation runner."""
    matrix_factorization = MatrixFactorizationConfig(
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        regularization=regularization,
        epochs=epochs,
        batch_size=batch_size,
    )
    config = config or ExperimentConfig()
    return _run_validation_model(
        processed_dir,
        model_name=BiasedMatrixFactorizationRecommender.model_name,
        model_factory=lambda user_count, item_count: BiasedMatrixFactorizationRecommender(
            user_count,
            item_count,
            latent_dim=matrix_factorization.latent_dim,
            learning_rate=matrix_factorization.learning_rate,
            regularization=matrix_factorization.regularization,
            epochs=matrix_factorization.epochs,
            batch_size=matrix_factorization.batch_size,
        ),
        model_configuration=matrix_factorization.as_dict(),
        run_name=f"matrix-factorization-validation-seed-{config.seed}",
        tracking_dir=tracking_dir,
        artifact_dir=artifact_dir,
        config=config,
        checkpoint_on_validation=True,
    )

def run_bpr_baseline(
    processed_dir: Path,
    *,
    latent_dim: int = 32,
    learning_rate: float = 0.01,
    regularization: float = 0.0001,
    epochs: int = 20,
    batch_size: int = 1024,
    negative_samples: int = 1,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run BPR through the common validation runner."""
    config = config or ExperimentConfig()
    bpr = BPRConfig(
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        regularization=regularization,
        epochs=epochs,
        batch_size=batch_size,
        negative_samples=negative_samples,
        positive_rating_threshold=config.positive_rating_threshold,
    )

    def make_model(user_count: int, item_count: int) -> RecommenderProtocol:
        return BPRRecommender(
            user_count,
            item_count,
            latent_dim=bpr.latent_dim,
            learning_rate=bpr.learning_rate,
            regularization=bpr.regularization,
            epochs=bpr.epochs,
            batch_size=bpr.batch_size,
            negative_samples=bpr.negative_samples,
            positive_rating_threshold=bpr.positive_rating_threshold,
        )

    return _run_validation_model(
        processed_dir,
        model_name=BPRRecommender.model_name,
        model_factory=make_model,
        model_configuration=bpr.as_dict(),
        run_name=f"bpr-validation-seed-{config.seed}",
        tracking_dir=tracking_dir,
        artifact_dir=artifact_dir,
        config=config,
        checkpoint_on_validation=True,
    )
