"""Tests for deterministic indexed tables and frozen chronological splits."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from recommender.data.download import DatasetError, DatasetSource, acquire_dataset
from recommender.data.prepare import prepare_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prepare" / "ml-1m"
RECORD_COUNTS = {"movies.dat": 5, "ratings.dat": 11, "users.dat": 3}
PROCESSED_FILES = (
    "manifest.json",
    "movies.parquet",
    "ratings.parquet",
    "test.parquet",
    "train.parquet",
    "users.parquet",
    "validation.parquet",
)


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


def _prepared(tmp_path: Path) -> tuple[Path, Path]:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    acquire_dataset(raw_dir, source=_source_archive(tmp_path))
    prepare_dataset(raw_dir, processed_dir)
    return raw_dir, processed_dir


def _rows(path: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def test_indices_and_splits_are_stable(tmp_path: Path) -> None:
    _, processed_dir = _prepared(tmp_path)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))

    assert [
        (row["movie_id"], row["item_idx"]) for row in _rows(processed_dir / "movies.parquet")
    ] == [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
    ]
    assert [
        (row["user_id"], row["user_idx"]) for row in _rows(processed_dir / "users.parquet")
    ] == [
        (1, 0),
        (2, 1),
        (3, 2),
    ]
    assert [
        (row["user_id"], row["movie_id"]) for row in _rows(processed_dir / "train.parquet")
    ] == [
        (1, 1),
        (1, 5),
        (1, 2),
        (2, 1),
    ]
    assert [
        (row["user_id"], row["movie_id"]) for row in _rows(processed_dir / "validation.parquet")
    ] == [
        (1, 3),
        (2, 3),
    ]
    assert [(row["user_id"], row["movie_id"]) for row in _rows(processed_dir / "test.parquet")] == [
        (1, 4),
        (2, 4),
    ]
    assert manifest["configuration"] == {
        "minimum_positive_history": 3,
        "positive_rating_threshold": 4,
    }
    assert manifest["counts"] == {
        "evaluated_users": 2,
        "excluded_users": 1,
        "movies": 5,
        "positive_ratings": 9,
        "ratings": 11,
        "unassigned_interactions": 3,
        "test_interactions": 2,
        "train_interactions": 4,
        "users": 3,
        "validation_interactions": 2,
    }
    assert manifest["time_ranges"] == {
        "ratings": {"max_timestamp": 400, "min_timestamp": 100},
        "test": {"max_timestamp": 400, "min_timestamp": 200},
        "train": {"max_timestamp": 200, "min_timestamp": 100},
        "validation": {"max_timestamp": 300, "min_timestamp": 100},
    }


def test_split_invariants_and_candidate_counts(tmp_path: Path) -> None:
    _, processed_dir = _prepared(tmp_path)
    train = _rows(processed_dir / "train.parquet")
    validation = _rows(processed_dir / "validation.parquet")
    test = _rows(processed_dir / "test.parquet")
    train_pairs = {(row["user_id"], row["movie_id"]) for row in train}
    validation_pairs = {(row["user_id"], row["movie_id"]) for row in validation}
    test_pairs = {(row["user_id"], row["movie_id"]) for row in test}

    assert train_pairs.isdisjoint(validation_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert validation_pairs.isdisjoint(test_pairs)
    assert all(row["split"] == "train" for row in train)
    assert any(row["rating"] == 3 for row in train)
    assert all(row["rating"] >= 4 and row["split"] == "validation" for row in validation)
    assert all(row["rating"] >= 4 and row["split"] == "test" for row in test)

    for user_id in (1, 2):
        user_train = [row for row in train if row["user_id"] == user_id]
        user_validation = next(row for row in validation if row["user_id"] == user_id)
        user_test = next(row for row in test if row["user_id"] == user_id)

        def order(row: dict[str, Any]) -> tuple[int, int, int]:
            return (int(row["timestamp"]), int(row["movie_id"]), int(row["interaction_idx"]))

        assert order(user_train[-1]) < order(user_validation) < order(user_test)
        assert user_validation["item_idx"] not in {row["item_idx"] for row in user_train}
        assert user_test["item_idx"] not in {row["item_idx"] for row in user_train}

    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {row["user_id"]: row["candidate_items"] for row in manifest["evaluated_users"]} == {
        1: 2,
        2: 4,
    }


def test_repeated_preparation_has_identical_artifacts(tmp_path: Path) -> None:
    raw_dir, first_dir = _prepared(tmp_path)
    second_dir = tmp_path / "processed-again"
    prepare_dataset(raw_dir, second_dir)

    for name in PROCESSED_FILES:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    assert prepare_dataset(raw_dir, first_dir) == first_dir / "manifest.json"


def test_stale_preparation_configuration_is_rejected(tmp_path: Path) -> None:
    raw_dir, processed_dir = _prepared(tmp_path)

    with pytest.raises(DatasetError, match="stale preparation configuration"):
        prepare_dataset(raw_dir, processed_dir, positive_rating_threshold=3)
