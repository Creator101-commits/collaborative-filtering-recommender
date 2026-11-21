"""Deterministic MovieLens indexing and chronological split preparation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from recommender.data.download import DEFAULT_RAW_DIR, MANIFEST_NAME, SOURCE_FILES, DatasetError
from recommender.data.source import (
    Movie,
    Rating,
    SourceDataError,
    User,
    parse_movie,
    parse_rating,
    parse_user,
)

DEFAULT_PROCESSED_DIR = Path("data/processed/movielens-1m")
PREPARATION_VERSION = 2
DEFAULT_POSITIVE_RATING_THRESHOLD = 4
DEFAULT_MINIMUM_POSITIVE_HISTORY = 3
PROCESSED_FILES = (
    "ratings.parquet",
    "movies.parquet",
    "users.parquet",
    "train.parquet",
    "train_for_test.parquet",
    "validation.parquet",
    "test.parquet",
)

_INTERACTION_FIELDS = (
    pa.field("interaction_idx", pa.int64(), nullable=False),
    pa.field("user_id", pa.int64(), nullable=False),
    pa.field("movie_id", pa.int64(), nullable=False),
    pa.field("rating", pa.int8(), nullable=False),
    pa.field("timestamp", pa.int64(), nullable=False),
    pa.field("user_idx", pa.int64(), nullable=False),
    pa.field("item_idx", pa.int64(), nullable=False),
)
_INTERACTION_SCHEMA = pa.schema(_INTERACTION_FIELDS)
_SPLIT_SCHEMA = pa.schema((*_INTERACTION_FIELDS, pa.field("split", pa.string(), nullable=False)))
_MOVIE_SCHEMA = pa.schema(
    (
        pa.field("movie_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("genres", pa.list_(pa.string()), nullable=False),
        pa.field("item_idx", pa.int64(), nullable=False),
    )
)
_USER_SCHEMA = pa.schema(
    (
        pa.field("user_id", pa.int64(), nullable=False),
        pa.field("gender", pa.string(), nullable=False),
        pa.field("age_group", pa.int8(), nullable=False),
        pa.field("occupation", pa.int8(), nullable=False),
        pa.field("zip_code", pa.string(), nullable=False),
        pa.field("user_idx", pa.int64(), nullable=False),
    )
)


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    """Configuration that is frozen into every processed-data manifest."""

    positive_rating_threshold: int = DEFAULT_POSITIVE_RATING_THRESHOLD
    minimum_positive_history: int = DEFAULT_MINIMUM_POSITIVE_HISTORY

    def __post_init__(self) -> None:
        if not 1 <= self.positive_rating_threshold <= 5:
            raise ValueError("positive_rating_threshold must be between 1 and 5")
        if self.minimum_positive_history < 3:
            raise ValueError("minimum_positive_history must be at least 3")

    def as_manifest(self) -> dict[str, int]:
        return {
            "minimum_positive_history": self.minimum_positive_history,
            "positive_rating_threshold": self.positive_rating_threshold,
        }


@dataclass(frozen=True, slots=True)
class _IndexedRating:
    interaction_idx: int
    user_id: int
    movie_id: int
    rating: int
    timestamp: int
    user_idx: int
    item_idx: int

    @property
    def order_key(self) -> tuple[int, int, int]:
        return (self.timestamp, self.movie_id, self.interaction_idx)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetError(f"manifest {path} must contain a JSON object")
    return cast(dict[str, object], value)


def _read_records[T: object](path: Path, parser: Callable[[str], T]) -> tuple[T, ...]:
    records: list[T] = []
    try:
        with path.open(encoding="latin-1", newline="") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    records.append(parser(line))
                except SourceDataError as error:
                    raise DatasetError(f"{path}:{line_number}: {error}") from error
    except OSError as error:
        raise DatasetError(f"cannot read source file {path}: {error}") from error
    if not records:
        raise DatasetError(f"{path} is empty")
    return tuple(records)


def _validate_raw_directory(raw_dir: Path) -> dict[str, object]:
    if not raw_dir.is_dir():
        raise DatasetError(f"raw dataset directory does not exist: {raw_dir}")

    manifest_path = raw_dir / MANIFEST_NAME
    manifest = _load_json_object(manifest_path)
    if manifest.get("dataset") != "MovieLens 1M" or manifest.get("manifest_version") != 1:
        raise DatasetError("raw manifest has an unsupported format")

    files_value = manifest.get("files")
    if not isinstance(files_value, dict) or set(files_value) != set(SOURCE_FILES):
        raise DatasetError("raw manifest does not describe the expected source files")
    files = cast(dict[str, object], files_value)

    for name in SOURCE_FILES:
        path = raw_dir / name
        if not path.is_file():
            raise DatasetError(f"raw dataset is missing {name}")
        entry_value = files.get(name)
        if not isinstance(entry_value, dict):
            raise DatasetError(f"raw manifest has invalid metadata for {name}")
        entry = cast(dict[str, object], entry_value)
        if entry.get("sha256") != _sha256(path) or entry.get("size_bytes") != path.stat().st_size:
            raise DatasetError(f"raw source file does not match its manifest: {name}")
    return manifest


def _load_source_data(
    raw_dir: Path, manifest: Mapping[str, object]
) -> tuple[tuple[User, ...], tuple[Movie, ...], tuple[Rating, ...]]:
    users = _read_records(raw_dir / "users.dat", parse_user)
    movies = _read_records(raw_dir / "movies.dat", parse_movie)
    ratings = _read_records(raw_dir / "ratings.dat", parse_rating)

    files_value = manifest["files"]
    files = cast(dict[str, object], files_value)
    for name, count in (
        ("users.dat", len(users)),
        ("movies.dat", len(movies)),
        ("ratings.dat", len(ratings)),
    ):
        entry = cast(dict[str, object], files[name])
        if entry.get("records") != count:
            raise DatasetError(f"{name} record count does not match its manifest")
    return users, movies, ratings


def _index_entities(
    users: tuple[User, ...], movies: tuple[Movie, ...]
) -> tuple[dict[int, int], dict[int, int]]:
    if len({user.user_id for user in users}) != len(users):
        raise DatasetError("users.dat contains duplicate user IDs")
    if len({movie.movie_id for movie in movies}) != len(movies):
        raise DatasetError("movies.dat contains duplicate movie IDs")

    user_index = {
        user.user_id: index
        for index, user in enumerate(sorted(users, key=lambda value: value.user_id))
    }
    item_index = {
        movie.movie_id: index
        for index, movie in enumerate(sorted(movies, key=lambda value: value.movie_id))
    }
    return user_index, item_index


def _index_ratings(
    ratings: tuple[Rating, ...], user_index: Mapping[int, int], item_index: Mapping[int, int]
) -> tuple[_IndexedRating, ...]:
    indexed: list[_IndexedRating] = []
    seen_pairs: set[tuple[int, int]] = set()
    for interaction_idx, rating in enumerate(ratings):
        if rating.user_id not in user_index:
            raise DatasetError(f"rating references unknown user ID {rating.user_id}")
        if rating.movie_id not in item_index:
            raise DatasetError(f"rating references unknown movie ID {rating.movie_id}")
        pair = (rating.user_id, rating.movie_id)
        if pair in seen_pairs:
            raise DatasetError(
                "ratings.dat contains duplicate user/movie interaction: "
                f"{rating.user_id}/{rating.movie_id}"
            )
        seen_pairs.add(pair)
        indexed.append(
            _IndexedRating(
                interaction_idx=interaction_idx,
                user_id=rating.user_id,
                movie_id=rating.movie_id,
                rating=rating.rating,
                timestamp=rating.timestamp,
                user_idx=user_index[rating.user_id],
                item_idx=item_index[rating.movie_id],
            )
        )
    return tuple(indexed)


def _split_ratings(
    ratings: tuple[_IndexedRating, ...],
    user_index: Mapping[int, int],
    item_count: int,
    config: PreparationConfig,
) -> tuple[dict[str, list[_IndexedRating]], int, list[dict[str, int]]]:
    ratings_by_user: defaultdict[int, list[_IndexedRating]] = defaultdict(list)
    positive_by_user: defaultdict[int, list[_IndexedRating]] = defaultdict(list)
    for rating in ratings:
        ratings_by_user[rating.user_idx].append(rating)
        if rating.rating >= config.positive_rating_threshold:
            positive_by_user[rating.user_idx].append(rating)

    partitions: dict[str, list[_IndexedRating]] = {
        "train": [],
        "train_for_test": [],
        "validation": [],
        "test": [],
    }
    excluded_users = 0
    evaluated_users: list[dict[str, int]] = []

    for user_id, user_idx in sorted(user_index.items(), key=lambda pair: pair[1]):
        user_ratings = sorted(
            positive_by_user.get(user_idx, []), key=lambda rating: rating.order_key
        )
        if len(user_ratings) < config.minimum_positive_history:
            excluded_users += 1
            continue

        validation = user_ratings[-2]
        test = user_ratings[-1]
        train = [
            rating
            for rating in sorted(
                ratings_by_user.get(user_idx, []), key=lambda rating: rating.order_key
            )
            if rating.order_key < validation.order_key
        ]
        train_for_test = [
            rating
            for rating in sorted(
                ratings_by_user.get(user_idx, []), key=lambda rating: rating.order_key
            )
            if rating.order_key < test.order_key
        ]
        if train[-1].order_key >= validation.order_key or validation.order_key >= test.order_key:
            raise DatasetError(f"chronological split ordering failed for user {user_id}")

        train_items = {rating.item_idx for rating in train}
        candidate_count = item_count - len(train_items)
        if candidate_count <= 0:
            raise DatasetError(f"user {user_id} has no eligible recommendation candidates")
        if validation.item_idx in train_items or test.item_idx in train_items:
            raise DatasetError(f"held-out item is present in training for user {user_id}")

        partitions["train"].extend(train)
        partitions["train_for_test"].extend(train_for_test)
        partitions["validation"].append(validation)
        partitions["test"].append(test)
        evaluated_users.append(
            {
                "candidate_items": candidate_count,
                "positive_history": len(user_ratings),
                "train_interactions": len(train),
                "user_id": user_id,
                "user_idx": user_idx,
            }
        )

    if not evaluated_users:
        raise DatasetError("no users meet the minimum positive-history requirement")
    return partitions, excluded_users, evaluated_users


def _interaction_columns(ratings: list[_IndexedRating]) -> dict[str, list[int]]:
    return {
        "interaction_idx": [rating.interaction_idx for rating in ratings],
        "user_id": [rating.user_id for rating in ratings],
        "movie_id": [rating.movie_id for rating in ratings],
        "rating": [rating.rating for rating in ratings],
        "timestamp": [rating.timestamp for rating in ratings],
        "user_idx": [rating.user_idx for rating in ratings],
        "item_idx": [rating.item_idx for rating in ratings],
    }


def _split_columns(ratings: list[_IndexedRating], split: str) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = _interaction_columns(ratings)
    columns["split"] = [split] * len(ratings)
    return columns


def _write_table(path: Path, columns: Mapping[str, list[Any]], schema: Any) -> None:
    table = pa.Table.from_pydict(dict(columns), schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=False)


def _schema_metadata(schema: Any) -> list[dict[str, object]]:
    return [
        {"name": field.name, "nullable": field.nullable, "type": str(field.type)}
        for field in schema
    ]


def _file_metadata(directory: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for name in PROCESSED_FILES:
        path = directory / name
        if not path.is_file():
            raise DatasetError(f"processed dataset is missing {name}")
        try:
            schema = pq.read_schema(path)
            parquet_metadata = pq.read_metadata(path)
        except (OSError, pa.ArrowException) as error:
            raise DatasetError(f"cannot inspect processed file {name}: {error}") from error
        metadata[name] = {
            "records": int(parquet_metadata.num_rows),
            "schema": _schema_metadata(schema),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return metadata


def _write_processed_tables(
    directory: Path,
    users: tuple[User, ...],
    movies: tuple[Movie, ...],
    ratings: tuple[_IndexedRating, ...],
    partitions: Mapping[str, list[_IndexedRating]],
) -> None:
    sorted_ratings = sorted(
        ratings,
        key=lambda rating: (
            rating.user_idx,
            rating.timestamp,
            rating.item_idx,
            rating.interaction_idx,
        ),
    )
    sorted_users = sorted(users, key=lambda user: user.user_id)
    sorted_movies = sorted(movies, key=lambda movie: movie.movie_id)

    _write_table(
        directory / "ratings.parquet", _interaction_columns(sorted_ratings), _INTERACTION_SCHEMA
    )
    _write_table(
        directory / "movies.parquet",
        {
            "movie_id": [movie.movie_id for movie in sorted_movies],
            "title": [movie.title for movie in sorted_movies],
            "genres": [list(movie.genres) for movie in sorted_movies],
            "item_idx": [index for index, movie in enumerate(sorted_movies)],
        },
        _MOVIE_SCHEMA,
    )
    _write_table(
        directory / "users.parquet",
        {
            "user_id": [user.user_id for user in sorted_users],
            "gender": [user.gender for user in sorted_users],
            "age_group": [user.age_group for user in sorted_users],
            "occupation": [user.occupation for user in sorted_users],
            "zip_code": [user.zip_code for user in sorted_users],
            "user_idx": [index for index, user in enumerate(sorted_users)],
        },
        _USER_SCHEMA,
    )
    for split in ("train", "train_for_test", "validation", "test"):
        split_ratings = sorted(
            partitions[split], key=lambda rating: (rating.user_idx, rating.order_key)
        )
        _write_table(
            directory / f"{split}.parquet", _split_columns(split_ratings, split), _SPLIT_SCHEMA
        )


def _time_range(ratings: Iterable[_IndexedRating]) -> dict[str, int]:
    timestamps = [rating.timestamp for rating in ratings]
    if not timestamps:
        raise DatasetError("cannot record a time range for an empty partition")
    return {"max_timestamp": max(timestamps), "min_timestamp": min(timestamps)}


def _build_manifest(
    raw_manifest_path: Path,
    config: PreparationConfig,
    users: tuple[User, ...],
    movies: tuple[Movie, ...],
    ratings: tuple[_IndexedRating, ...],
    partitions: Mapping[str, list[_IndexedRating]],
    excluded_users: int,
    evaluated_users: list[dict[str, int]],
) -> dict[str, object]:
    positive_count = sum(rating.rating >= config.positive_rating_threshold for rating in ratings)
    return {
        "configuration": config.as_manifest(),
        "counts": {
            "evaluated_users": len(evaluated_users),
            "excluded_users": excluded_users,
            "movies": len(movies),
            "positive_ratings": positive_count,
            "ratings": len(ratings),
            "unassigned_interactions": len(ratings)
            - sum(len(partitions[split]) for split in ("train", "validation", "test")),
            "test_interactions": len(partitions["test"]),
            "train_for_test_interactions": len(partitions["train_for_test"]),
            "train_interactions": len(partitions["train"]),
            "users": len(users),
            "validation_interactions": len(partitions["validation"]),
        },
        "dataset": "MovieLens 1M",
        "evaluated_users": evaluated_users,
        "manifest_version": 1,
        "preparation_version": PREPARATION_VERSION,
        "raw_manifest_sha256": _sha256(raw_manifest_path),
        "split_protocol": {
            "candidate_catalog": "all indexed movies except each user's training items",
            "chronological_order": ["timestamp", "movie_id", "interaction_idx"],
            "held_out_events": {
                "test": "last positive interaction",
                "validation": "second-last positive interaction",
            },
            "positive_interactions_only": False,
            "training_events": "all source interactions earlier than validation",
            "test_training_events": "all source interactions earlier than test",
        },
        "time_ranges": {
            "ratings": _time_range(ratings),
            "test": _time_range(partitions["test"]),
            "train": _time_range(partitions["train"]),
            "train_for_test": _time_range(partitions["train_for_test"]),
            "validation": _time_range(partitions["validation"]),
        },
    }


def _validate_existing(directory: Path, raw_manifest_path: Path, config: PreparationConfig) -> None:
    if not directory.is_dir():
        raise DatasetError(f"processed output exists but is not a directory: {directory}")
    manifest = _load_json_object(directory / MANIFEST_NAME)
    if (
        manifest.get("dataset") != "MovieLens 1M"
        or manifest.get("manifest_version") != 1
        or manifest.get("preparation_version") != PREPARATION_VERSION
    ):
        raise DatasetError("existing processed output has an unsupported or stale manifest")
    if manifest.get("configuration") != config.as_manifest():
        raise DatasetError("existing processed output has a stale preparation configuration")
    if manifest.get("raw_manifest_sha256") != _sha256(raw_manifest_path):
        raise DatasetError("existing processed output references a different raw manifest")

    files_value = manifest.get("files")
    if not isinstance(files_value, dict) or set(files_value) != set(PROCESSED_FILES):
        raise DatasetError("existing processed manifest has incomplete file metadata")
    if cast(dict[str, object], files_value) != _file_metadata(directory):
        raise DatasetError("existing processed files do not match their manifest")


def prepare_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    positive_rating_threshold: int = DEFAULT_POSITIVE_RATING_THRESHOLD,
    minimum_positive_history: int = DEFAULT_MINIMUM_POSITIVE_HISTORY,
) -> Path:
    """Create deterministic indexed Parquet tables and frozen chronological splits."""
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    if raw_dir == output_dir:
        raise DatasetError("raw and processed directories must be different")

    config = PreparationConfig(
        positive_rating_threshold=positive_rating_threshold,
        minimum_positive_history=minimum_positive_history,
    )
    raw_manifest = _validate_raw_directory(raw_dir)
    raw_manifest_path = raw_dir / MANIFEST_NAME
    if output_dir.exists():
        _validate_existing(output_dir, raw_manifest_path, config)
        return output_dir / MANIFEST_NAME

    users, movies, source_ratings = _load_source_data(raw_dir, raw_manifest)
    user_index, item_index = _index_entities(users, movies)
    ratings = _index_ratings(source_ratings, user_index, item_index)
    partitions, excluded_users, evaluated_users = _split_ratings(
        ratings, user_index, len(item_index), config
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".movielens-processed-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "processed"
        staging.mkdir()
        _write_processed_tables(staging, users, movies, ratings, partitions)
        manifest = _build_manifest(
            raw_manifest_path,
            config,
            users,
            movies,
            ratings,
            partitions,
            excluded_users,
            evaluated_users,
        )
        manifest["files"] = _file_metadata(staging)
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            staging.rename(output_dir)
        except FileExistsError as error:
            raise DatasetError(f"{output_dir} was created by another process") from error
    return output_dir / MANIFEST_NAME
