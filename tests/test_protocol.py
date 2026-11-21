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


def test_audit_requires_all_models_and_test_table_for_split_invariants(
    tmp_path: Path,
) -> None:
    processed_dir = _prepared(tmp_path)
    configurations: dict[str, dict[str, object]] = {name: {} for name in FINAL_MODEL_NAMES}
    audit = audit_comparison_inputs(processed_dir, configurations)

    assert audit.item_count == 5
    assert audit.evaluated_users == 2
    assert len(audit.candidate_catalog_sha256) == 64

    with pytest.raises(ProtocolError, match="model configurations"):
        audit_comparison_inputs(processed_dir, {"popularity": {}})

    (processed_dir / "test.parquet").unlink()
    with pytest.raises(ProtocolError, match="test.parquet"):
        audit_comparison_inputs(processed_dir, configurations)


def test_experiment_card_is_deterministic_and_not_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "experiment-card.json"
    configurations: dict[str, dict[str, object]] = {
        "popularity": {},
        "user_knn": {"similarity": "cosine"},
        "item_knn": {"similarity": "cosine"},
        "matrix_factorization": {"latent_dim": 32, "epochs": 7},
        "bpr": {"latent_dim": 32, "epochs": 9},
    }

    written = write_experiment_card(path, "a" * 64, configurations, git_revision="abc")
    assert written == path
    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["data_manifest_sha256"] == "a" * 64
    assert set(card["models"]) == set(FINAL_MODEL_NAMES)

    assert write_experiment_card(path, "a" * 64, configurations, git_revision="abc") == path
    with pytest.raises(ProtocolError, match="already exists"):
        write_experiment_card(path, "b" * 64, configurations, git_revision="abc")


def _interaction(user: int, timestamp: int, index: int) -> dict[str, object]:
    return {"user_idx": user, "timestamp": timestamp, "interaction_idx": index}


def test_train_for_test_audit_accepts_leave_one_out_history() -> None:
    from recommender.experiments.protocol import _audit_train_for_test

    train = [_interaction(0, 1, 0)]
    train_for_test = [_interaction(0, 1, 0), _interaction(0, 2, 1), _interaction(0, 3, 2)]
    validation = [_interaction(0, 2, 1)]
    test = [_interaction(0, 4, 3)]

    _audit_train_for_test(train, train_for_test, validation, test)


def test_train_for_test_audit_rejects_missing_training_row() -> None:
    from recommender.experiments.protocol import _audit_train_for_test

    with pytest.raises(ProtocolError, match="every training interaction"):
        _audit_train_for_test(
            [_interaction(0, 1, 0), _interaction(0, 2, 1)],
            [_interaction(0, 1, 0)],
            [_interaction(0, 3, 2)],
            [_interaction(0, 4, 3)],
        )


def test_train_for_test_audit_rejects_missing_validation_row() -> None:
    from recommender.experiments.protocol import _audit_train_for_test

    with pytest.raises(ProtocolError, match="validation interaction"):
        _audit_train_for_test(
            [_interaction(0, 1, 0)],
            [_interaction(0, 1, 0), _interaction(0, 3, 2)],
            [_interaction(0, 2, 1)],
            [_interaction(0, 4, 3)],
        )


def test_train_for_test_audit_rejects_test_row() -> None:
    from recommender.experiments.protocol import _audit_train_for_test

    with pytest.raises(ProtocolError, match="test interaction"):
        _audit_train_for_test(
            [_interaction(0, 1, 0)],
            [_interaction(0, 1, 0), _interaction(0, 2, 1), _interaction(0, 4, 3)],
            [_interaction(0, 2, 1)],
            [_interaction(0, 4, 3)],
        )


def test_train_for_test_audit_rejects_late_interaction() -> None:
    from recommender.experiments.protocol import _audit_train_for_test

    with pytest.raises(ProtocolError, match="later than the test interaction"):
        _audit_train_for_test(
            [_interaction(0, 1, 0)],
            [_interaction(0, 1, 0), _interaction(0, 2, 1), _interaction(0, 9, 2)],
            [_interaction(0, 2, 1)],
            [_interaction(0, 4, 3)],
        )


def test_select_configuration_prefers_ndcg_then_recall_then_speed() -> None:
    from recommender.experiments.protocol import select_configuration

    candidates = [
        {"ndcg@10": 0.5, "recall@10": 0.9, "train_seconds": 1.0, "name": "a"},
        {"ndcg@10": 0.6, "recall@10": 0.1, "train_seconds": 9.0, "name": "b"},
        {"ndcg@10": 0.6, "recall@10": 0.2, "train_seconds": 9.0, "name": "c"},
        {"ndcg@10": 0.6, "recall@10": 0.2, "train_seconds": 3.0, "name": "d"},
    ]
    assert select_configuration(candidates)["name"] == "d"


def test_select_configuration_rejects_empty_and_missing_metrics() -> None:
    from recommender.experiments.protocol import select_configuration

    with pytest.raises(ProtocolError, match="no validation candidates"):
        select_configuration([])
    with pytest.raises(ProtocolError, match="selection metrics"):
        select_configuration([{"name": "a"}])


def test_build_model_configuration_freezes_checkpoint_epoch() -> None:
    from recommender.experiments.protocol import build_model_configuration

    row = {
        "latent_dim": 32,
        "learning_rate": 0.01,
        "regularization": 0.001,
        "ndcg@10": 0.5,
        "recall@10": 0.7,
        "train_seconds": 3.0,
        "model": "matrix_factorization",
    }
    config = build_model_configuration("matrix_factorization", row, best_epoch=7)
    assert config == {
        "latent_dim": 32,
        "learning_rate": 0.01,
        "regularization": 0.001,
        "epochs": 7,
    }
    assert not any("@" in key for key in config)


def test_build_model_configuration_requires_epoch_for_learned() -> None:
    from recommender.experiments.protocol import build_model_configuration

    with pytest.raises(ProtocolError, match="checkpoint epoch"):
        build_model_configuration("bpr", {"latent_dim": 16}, best_epoch=None)


def test_build_model_configuration_omits_epochs_for_baselines() -> None:
    from recommender.experiments.protocol import build_model_configuration

    config = build_model_configuration(
        "user_knn",
        {
            "n_neighbors": 40,
            "similarity": "cosine",
            "shrinkage": 10.0,
            "min_support": 1,
            "ndcg@10": 0.4,
        },
        best_epoch=None,
    )
    assert config == {
        "n_neighbors": 40,
        "similarity": "cosine",
        "shrinkage": 10.0,
        "min_support": 1,
    }


def test_experiment_card_rejects_unknown_config_keys(tmp_path: Path) -> None:
    path = tmp_path / "card.json"
    configurations: dict[str, dict[str, object]] = {
        name: {} for name in FINAL_MODEL_NAMES
    }
    configurations["bpr"] = {"latent_dim": 16, "ndcg@10": 0.9}
    with pytest.raises(ProtocolError, match="unknown keys"):
        write_experiment_card(
            path, "0" * 64, configurations, git_revision="abc123"
        )
