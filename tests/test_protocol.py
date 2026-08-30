from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommender.data.download import DatasetSource, acquire_dataset
from recommender.data.prepare import prepare_dataset
from recommender.experiments.protocol import (
    FINAL_MODEL_NAMES,
    ComparisonProtocol,
    ProtocolError,
    audit_comparison_inputs,
    write_experiment_card,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prepare" / "ml-1m"


def _prepared(tmp_path: Path) -> Path:
    archive = tmp_path / "ml-1m.zip"
    import hashlib
    import zipfile

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


def test_protocol_freezes_metrics_slices_seeds_and_budget() -> None:
    protocol = ComparisonProtocol()

    assert protocol.primary_cutoff == 10
    assert protocol.metric_cutoffs == (5, 10, 20)
    assert protocol.learned_model_seeds == (0, 1, 2)
    assert protocol.bootstrap_replicates == 10_000
    assert protocol.as_dict()["novelty"] == "laplace_smoothed_self_information_bits"


def test_protocol_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="primary_cutoff"):
        ComparisonProtocol(metric_cutoffs=(5, 20), primary_cutoff=10)
    with pytest.raises(ValueError, match="confidence_level"):
        ComparisonProtocol(confidence_level=1.0)


def test_audit_requires_all_models_and_does_not_read_test_labels(tmp_path: Path) -> None:
    processed_dir = _prepared(tmp_path)
    (processed_dir / "test.parquet").unlink()

    configurations: dict[str, dict[str, object]] = {name: {} for name in FINAL_MODEL_NAMES}
    audit = audit_comparison_inputs(processed_dir, configurations)

    assert audit.item_count == 5
    assert audit.evaluated_users == 2
    assert len(audit.candidate_catalog_sha256) == 64

    with pytest.raises(ProtocolError, match="model configurations"):
        audit_comparison_inputs(processed_dir, {"popularity": {}})


def test_experiment_card_is_deterministic_and_not_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "experiment-card.json"
    configurations: dict[str, dict[str, object]] = {
        name: {"selected_by": "validation"} for name in FINAL_MODEL_NAMES
    }

    written = write_experiment_card(path, "a" * 64, configurations, git_revision="abc")
    assert written == path
    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["data_manifest_sha256"] == "a" * 64
    assert set(card["models"]) == set(FINAL_MODEL_NAMES)

    assert write_experiment_card(path, "a" * 64, configurations, git_revision="abc") == path
    with pytest.raises(ProtocolError, match="already exists"):
        write_experiment_card(path, "b" * 64, configurations, git_revision="abc")
