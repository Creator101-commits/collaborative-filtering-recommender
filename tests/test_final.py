from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

from recommender.data.download import DatasetSource, acquire_dataset
from recommender.data.prepare import prepare_dataset
from recommender.experiments import (
    FINAL_MODEL_NAMES,
    ComparisonProtocol,
    run_final_comparison,
    write_experiment_card,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prepare" / "ml-1m"


def _prepared(tmp_path: Path) -> Path:
    archive = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for fixture in FIXTURE_DIR.iterdir():
            output.write(fixture, f"ml-1m/{fixture.name}")
    content = archive.read_bytes()
    source = DatasetSource(
        url=archive.as_uri(),
        archive_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        md5=hashlib.md5(content).hexdigest(),  # noqa: S324
        record_counts={"movies.dat": 5, "ratings.dat": 11, "users.dat": 3},
    )
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    acquire_dataset(raw_dir, source=source)
    prepare_dataset(raw_dir, processed_dir)
    return processed_dir


def test_final_comparison_writes_test_metrics_and_paired_analysis(tmp_path: Path) -> None:
    processed_dir = _prepared(tmp_path)
    manifest = processed_dir / "manifest.json"
    configurations: dict[str, dict[str, object]] = {
        "popularity": {},
        "user_knn": {"n_neighbors": 1, "similarity": "cosine", "shrinkage": 0.0, "min_support": 1},
        "item_knn": {"n_neighbors": 1, "similarity": "cosine", "shrinkage": 0.0, "min_support": 1},
        "matrix_factorization": {"latent_dim": 2, "epochs": 1, "batch_size": 4},
        "bpr": {"latent_dim": 2, "epochs": 1, "batch_size": 4},
    }
    protocol = ComparisonProtocol(bootstrap_replicates=20)
    card = tmp_path / "experiment-card.json"
    write_experiment_card(
        card,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        configurations,
        git_revision="test-revision",
        protocol=protocol,
    )

    result = run_final_comparison(
        processed_dir,
        card,
        tracking_dir=tmp_path / "mlruns",
        artifact_dir=tmp_path / "final",
        protocol=protocol,
    )

    assert result.leading_model in FINAL_MODEL_NAMES
    assert result.results_path.is_file()
    assert result.paired_results_path.is_file()
    assert result.slices_path.is_file()
    with result.results_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 9
    assert {row["split"] for row in rows} == {"test"}
    assert all(row["protocol_version"] == "phase-8-v1" for row in rows)
