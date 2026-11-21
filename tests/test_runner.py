"""Tests for the validation-only MLflow experiment runner."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from recommender.data.download import DatasetSource, acquire_dataset
from recommender.data.prepare import prepare_dataset
from recommender.experiments import (
    PROTOCOL_VERSION,
    ExperimentConfig,
    run_bpr_baseline,
    run_matrix_factorization_baseline,
    run_neighborhood_baseline,
    run_popularity_baseline,
)
from recommender.models import BPRRecommender, PopularityRecommender

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prepare" / "ml-1m"
RECORD_COUNTS = {"movies.dat": 5, "ratings.dat": 11, "users.dat": 3}


def _source_archive(tmp_path: Path) -> DatasetSource:
    archive = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for fixture in FIXTURE_DIR.iterdir():
            output.write(fixture, f"ml-1m/{fixture.name}")
    content = archive.read_bytes()
    return DatasetSource(
        url=archive.as_uri(),
        archive_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        md5=hashlib.md5(content).hexdigest(),  # noqa: S324
        record_counts=RECORD_COUNTS,
    )


def _prepared(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    acquire_dataset(raw_dir, source=_source_archive(tmp_path))
    prepare_dataset(raw_dir, processed_dir)
    return processed_dir


def test_runner_logs_validation_run_and_artifacts(tmp_path: Path) -> None:
    processed_dir = _prepared(tmp_path)

    result = run_popularity_baseline(
        processed_dir,
        tracking_dir=tmp_path / "mlruns",
        artifact_dir=tmp_path / "artifacts",
        config=ExperimentConfig(cutoff=2, seed=11, experiment_name="test-baseline"),
    )

    manifest_path = processed_dir / "manifest.json"
    assert result.data_manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert result.model_path.is_file()
    assert result.results_path.is_file()
    assert result.metrics["recall@2"] >= 0
    assert result.metrics["catalog_coverage@2"] > 0

    restored = PopularityRecommender.load(result.model_path)
    assert restored.metadata is not None
    assert restored.metadata.data_manifest_sha256 == result.data_manifest_sha256
    assert restored.metadata.seed == 11

    with result.results_path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert row["split"] == "validation"
    assert row["cutoff"] == "2"
    assert row["protocol_version"] == PROTOCOL_VERSION
    assert row["positive_rating_threshold"] == "4"
    assert "test" not in row

    run_files = list((tmp_path / "mlruns").rglob("meta.yaml"))
    assert run_files


def test_runner_does_not_need_test_table(tmp_path: Path) -> None:
    processed_dir = _prepared(tmp_path)
    (processed_dir / "test.parquet").unlink()

    result = run_popularity_baseline(
        processed_dir,
        tracking_dir=tmp_path / "mlruns",
        artifact_dir=tmp_path / "artifacts",
    )

    assert result.results_path.is_file()
    metadata = json.loads(result.model_path.read_text(encoding="utf-8"))["metadata"]
    assert metadata["model_name"] == "popularity"


@pytest.mark.parametrize("model", ["user_knn", "item_knn"])
@pytest.mark.parametrize("similarity", ["cosine", "adjusted_cosine"])
def test_neighborhood_runner_saves_selected_configuration(
    model: str, similarity: str, tmp_path: Path
) -> None:
    processed_dir = _prepared(tmp_path)
    result = run_neighborhood_baseline(
        processed_dir,
        model=model,
        n_neighbors=2,
        similarity=similarity,
        shrinkage=0.0,
        min_support=1,
        tracking_dir=tmp_path / "mlruns",
        artifact_dir=tmp_path / "artifacts",
        config=ExperimentConfig(
            cutoff=2, seed=5, experiment_name=f"test-{model}-{similarity}"
        ),
    )

    assert result.model_path.is_file()
    metadata = json.loads(result.model_path.read_text(encoding="utf-8"))["metadata"]
    assert metadata["model_name"] == model
    assert metadata["configuration"]["n_neighbors"] == 2
    assert metadata["configuration"]["min_support"] == 1
    with result.results_path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert row["model"] == model
    assert row["n_neighbors"] == "2"
    assert row["similarity"] == similarity
    assert row["shrinkage"] == "0.0"


def test_runner_matrix_factorization_uses_validation_checkpoint(tmp_path: Path) -> None:
    processed_dir = _prepared(tmp_path)
    result = run_matrix_factorization_baseline(
        processed_dir,
        latent_dim=2,
        epochs=3,
        batch_size=4,
        tracking_dir=tmp_path / "mlruns",
        artifact_dir=tmp_path / "artifacts",
        config=ExperimentConfig(cutoff=2, seed=9, experiment_name="test-mf"),
    )

    assert result.model_path.suffix == ".pt"
    assert result.model_path.is_file()
    assert result.results_path.is_file()
    assert result.metrics["recall@2"] >= 0

def test_runner_bpr_uses_validation_checkpoint(tmp_path: Path) -> None:
    processed_dir = _prepared(tmp_path)
    result = run_bpr_baseline(
        processed_dir,
        latent_dim=2,
        epochs=3,
        batch_size=4,
        tracking_dir=tmp_path / "mlruns",
        artifact_dir=tmp_path / "artifacts",
        config=ExperimentConfig(cutoff=2, seed=9, experiment_name="test-bpr"),
    )

    assert result.model_path.suffix == ".pt"
    assert result.model_path.is_file()
    assert result.results_path.is_file()
    restored = BPRRecommender.load(result.model_path)
    assert restored.metadata is not None
    assert restored.metadata.model_name == "bpr"
    assert restored.best_epoch is not None
