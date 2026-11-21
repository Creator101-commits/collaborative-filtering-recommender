"""Sparse user- and item-neighborhood recommenders."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, cast

import scipy.sparse as sparse  # type: ignore[import-untyped]

from recommender.models.artifacts import ARTIFACT_VERSION, ArtifactMetadata
from recommender.models.protocol import Interaction

NeighborhoodOrientation = Literal["user", "item"]


@dataclass(frozen=True, slots=True)
class NeighborhoodConfig:
    """Validation-controlled settings for a sparse neighborhood model."""

    n_neighbors: int = 40
    similarity: str = "cosine"
    shrinkage: float = 10.0
    min_support: int = 1
    aggregation: str = "weighted_mean"

    def __post_init__(self) -> None:
        if not isinstance(self.n_neighbors, int) or isinstance(self.n_neighbors, bool):
            raise ValueError("n_neighbors must be an integer")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if self.similarity not in {"cosine", "adjusted_cosine"}:
            raise ValueError("similarity must be cosine or adjusted_cosine")
        if (
            isinstance(self.shrinkage, bool)
            or not isinstance(self.shrinkage, (int, float))
            or not math.isfinite(float(self.shrinkage))
            or self.shrinkage < 0
        ):
            raise ValueError("shrinkage must be a finite non-negative number")
        if not isinstance(self.min_support, int) or isinstance(self.min_support, bool):
            raise ValueError("min_support must be an integer")
        if self.min_support <= 0:
            raise ValueError("min_support must be positive")
        if self.aggregation not in {"weighted_mean", "weighted_sum"}:
            raise ValueError("aggregation must be weighted_mean or weighted_sum")

    def as_dict(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation,
            "min_support": self.min_support,
            "n_neighbors": self.n_neighbors,
            "shrinkage": self.shrinkage,
            "similarity": self.similarity,
        }


def _integer_field(row: Mapping[str, Any], field: str) -> int:
    try:
        value = row[field]
        result = int(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"interaction has no valid {field}") from error
    if isinstance(value, bool) or result != value:
        raise ValueError(f"interaction has no valid {field}")
    return result


def _training_matrix(
    interactions: Iterable[Interaction], user_count: int, item_count: int
) -> sparse.csr_matrix:
    users: list[int] = []
    items: list[int] = []
    ratings: list[float] = []
    seen: set[tuple[int, int]] = set()
    for row in interactions:
        user_idx = _integer_field(row, "user_idx")
        item_idx = _integer_field(row, "item_idx")
        if not 0 <= user_idx < user_count:
            raise ValueError(f"user_idx {user_idx} is outside the user catalog")
        if not 0 <= item_idx < item_count:
            raise ValueError(f"item_idx {item_idx} is outside the item catalog")
        pair = (user_idx, item_idx)
        if pair in seen:
            raise ValueError("training interactions contain duplicate user/item pairs")
        seen.add(pair)
        value = row.get("rating", 1.0)
        try:
            rating = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("training interaction has an invalid rating") from error
        if not math.isfinite(rating) or rating <= 0:
            raise ValueError("training interaction rating must be positive and finite")
        users.append(user_idx)
        items.append(item_idx)
        ratings.append(rating)
    if not ratings:
        raise ValueError("training interactions must not be empty")
    matrix = sparse.coo_matrix(
        (ratings, (users, items)), shape=(user_count, item_count), dtype=float
    ).tocsr()
    matrix.sort_indices()
    return matrix


def _normalise_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    squared_norms = matrix.multiply(matrix).sum(axis=1).tolist()
    inverse_norms = []
    for row in squared_norms:
        value = float(row[0])
        inverse_norms.append(1.0 / math.sqrt(value) if value else 0.0)
    return sparse.diags(inverse_norms).dot(matrix).tocsr()


def _adjusted_values(
    values: sparse.csr_matrix,
    presence: sparse.csr_matrix,
    orientation: NeighborhoodOrientation,
) -> sparse.csr_matrix:
    centered = values.copy()
    if orientation == "user":
        sums = values.sum(axis=1).tolist()
        counts = presence.sum(axis=1).tolist()
        means = [
            float(row[0]) / float(count[0]) if count[0] else 0.0 for row, count in zip(sums, counts)
        ]
        for row_index in range(centered.shape[0]):
            start, stop = centered.indptr[row_index : row_index + 2]
            centered.data[start:stop] -= means[row_index]
    else:
        sums = values.sum(axis=0).tolist()[0]
        counts = presence.sum(axis=0).tolist()[0]
        means = [
            float(total) / float(count) if count else 0.0 for total, count in zip(sums, counts)
        ]
        for row_index in range(centered.shape[0]):
            start, stop = centered.indptr[row_index : row_index + 2]
            for position in range(start, stop):
                centered.data[position] -= means[centered.indices[position]]
    return centered


def _neighbor_matrix(
    values: sparse.csr_matrix,
    presence: sparse.csr_matrix,
    config: NeighborhoodConfig,
    orientation: NeighborhoodOrientation,
) -> sparse.csr_matrix:
    entity_count = values.shape[0]
    similarity_values = (
        _adjusted_values(values, presence, orientation)
        if config.similarity == "adjusted_cosine"
        else values
    )
    normalized = _normalise_rows(similarity_values)
    neighbors: list[list[int]] = []
    similarities: list[list[float]] = []
    block_size = 256
    for start in range(0, entity_count, block_size):
        stop = min(start + block_size, entity_count)
        similarity_block = normalized[start:stop].dot(normalized.T)
        support_block = presence[start:stop].dot(presence.T)
        for local_index in range(stop - start):
            entity_idx = start + local_index
            similarity_row = similarity_block.getrow(local_index)
            support_row = support_block.getrow(local_index)
            supports = {
                int(index): int(round(float(value)))
                for index, value in zip(support_row.indices, support_row.data)
            }
            row: list[tuple[int, float]] = []
            for index, value in zip(similarity_row.indices, similarity_row.data):
                neighbor_idx = int(index)
                support = supports.get(neighbor_idx, 0)
                if neighbor_idx == entity_idx or support < config.min_support:
                    continue
                similarity = float(value)
                if similarity == 0:
                    continue
                if config.shrinkage:
                    similarity *= support / (support + config.shrinkage)
                row.append((neighbor_idx, similarity))
            row.sort(key=lambda pair: (-pair[1], pair[0]))
            selected = row[: config.n_neighbors]
            neighbors.append([index for index, _ in selected])
            similarities.append([value for _, value in selected])

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row_indices, row_values in zip(neighbors, similarities):
        indices.extend(row_indices)
        data.extend(row_values)
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (data, indices, indptr), shape=(entity_count, entity_count), dtype=float
    )


def _matrix_payload(matrix: sparse.csr_matrix) -> dict[str, object]:
    return {
        "data": [float(value) for value in matrix.data],
        "indices": [int(value) for value in matrix.indices],
        "indptr": [int(value) for value in matrix.indptr],
        "shape": [int(value) for value in matrix.shape],
    }


def _integer_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"artifact {label} must be a list of integers")
    return [int(item) for item in value]


def _number_list(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"artifact {label} must be a list of numbers")
    return [float(item) for item in value]


def _matrix_from_payload(value: object, label: str) -> sparse.csr_matrix:
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact {label} must be an object")
    shape = _integer_list(value.get("shape"), f"{label}.shape")
    indptr = _integer_list(value.get("indptr"), f"{label}.indptr")
    indices = _integer_list(value.get("indices"), f"{label}.indices")
    data = _number_list(value.get("data"), f"{label}.data")
    if len(shape) != 2 or any(dimension < 0 for dimension in shape):
        raise ValueError(f"artifact {label} has an invalid shape")
    if len(indptr) != shape[0] + 1 or len(indices) != len(data):
        raise ValueError(f"artifact {label} has inconsistent sparse arrays")
    if not indptr or indptr[0] != 0 or indptr[-1] != len(indices):
        raise ValueError(f"artifact {label} has invalid row pointers")
    if any(left > right for left, right in zip(indptr, indptr[1:])):
        raise ValueError(f"artifact {label} has invalid row pointers")
    if any(index < 0 or index >= shape[1] for index in indices):
        raise ValueError(f"artifact {label} has an out-of-range index")
    return sparse.csr_matrix((data, indices, indptr), shape=tuple(shape), dtype=float)


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"artifact {label} must be a positive integer")
    return value


class _NeighborhoodRecommender:
    """Shared sparse fitting, scoring, and artifact behavior for both KNN axes."""

    model_name: ClassVar[str]
    orientation: ClassVar[NeighborhoodOrientation]

    def __init__(
        self,
        user_count: int,
        item_count: int,
        *,
        n_neighbors: int = 40,
        similarity: str = "cosine",
        shrinkage: float = 10.0,
        min_support: int = 1,
        aggregation: str = "weighted_mean",
    ) -> None:
        self.user_count = _positive_integer(user_count, "user_count")
        self.item_count = _positive_integer(item_count, "item_count")
        self.config = NeighborhoodConfig(
            n_neighbors=n_neighbors,
            similarity=similarity,
            shrinkage=shrinkage,
            min_support=min_support,
            aggregation=aggregation,
        )
        self._entity_matrix: sparse.csr_matrix | None = None
        self._presence_matrix: sparse.csr_matrix | None = None
        self._neighbors: sparse.csr_matrix | None = None
        self.metadata: ArtifactMetadata | None = None
        self._fitted = False

    @property
    def neighbor_indices(self) -> dict[int, tuple[int, ...]]:
        self._check_fitted()
        assert self._neighbors is not None
        return {
            row: tuple(int(index) for index in self._neighbors.getrow(row).indices)
            for row in range(self._neighbors.shape[0])
        }

    @property
    def neighbor_similarities(self) -> dict[int, tuple[float, ...]]:
        self._check_fitted()
        assert self._neighbors is not None
        return {
            row: tuple(float(value) for value in self._neighbors.getrow(row).data)
            for row in range(self._neighbors.shape[0])
        }

    def fit(self, interactions: Iterable[Interaction], *, seed: int = 0) -> Self:
        del seed
        matrix = _training_matrix(interactions, self.user_count, self.item_count)
        if self.orientation == "user":
            entity_matrix = matrix
        else:
            entity_matrix = matrix.T.tocsr()
        presence = entity_matrix.copy()
        presence.data.fill(1.0)
        self._entity_matrix = entity_matrix
        self._presence_matrix = presence
        self._neighbors = _neighbor_matrix(entity_matrix, presence, self.config, self.orientation)
        self.metadata = None
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("recommender must be fitted before use")

    def _score_vector(self, user_idx: int) -> list[float]:
        self._check_fitted()
        if not 0 <= user_idx < self.user_count:
            raise ValueError(f"user_idx {user_idx} is outside the user catalog")
        assert self._entity_matrix is not None
        assert self._presence_matrix is not None
        assert self._neighbors is not None
        weighted_mean = self.config.aggregation == "weighted_mean"
        if self.orientation == "user":
            weights = self._neighbors.getrow(user_idx)
            numerator = weights.dot(self._entity_matrix)
            denominator = (
                abs(weights).dot(self._presence_matrix) if weighted_mean else None
            )
        else:
            ratings = self._entity_matrix.getcol(user_idx)
            presence = self._presence_matrix.getcol(user_idx)
            numerator = self._neighbors.dot(ratings)
            denominator = (
                abs(self._neighbors).dot(presence) if weighted_mean else None
            )
        numerator_values = numerator.toarray().ravel()
        if not weighted_mean:
            return [float(value) for value in numerator_values]
        assert denominator is not None
        denominator_values = denominator.toarray().ravel()
        return [
            float(value) / float(normalizer) if normalizer else 0.0
            for value, normalizer in zip(numerator_values, denominator_values)
        ]

    def score(self, user_idx: int, item_idx: int) -> float:
        if not 0 <= item_idx < self.item_count:
            raise ValueError(f"item_idx {item_idx} is outside the item catalog")
        return self._score_vector(user_idx)[item_idx]

    def recommend(self, user_idx: int, candidates: Iterable[int], cutoff: int = 10) -> list[int]:
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        candidate_set: set[int] = set()
        for item_idx in candidates:
            if not 0 <= item_idx < self.item_count:
                raise ValueError(f"item_idx {item_idx} is outside the item catalog")
            candidate_set.add(item_idx)
        scores = self._score_vector(user_idx)
        assert self._entity_matrix is not None
        seen_items = (
            self._entity_matrix.getrow(user_idx).indices
            if self.orientation == "user"
            else self._entity_matrix.getcol(user_idx).tocoo().row
        )
        candidate_set.difference_update(int(item_idx) for item_idx in seen_items)
        return sorted(candidate_set, key=lambda item: (-scores[item], item))[:cutoff]

    def _default_metadata(self) -> ArtifactMetadata:
        return ArtifactMetadata(
            model_name=self.model_name,
            data_manifest_sha256="",
            configuration={"neighborhood": self.config.as_dict()},
            seed=0,
            git_revision="unknown",
        )

    def save(self, path: Path, *, metadata: ArtifactMetadata | None = None) -> Path:
        self._check_fitted()
        assert self._entity_matrix is not None
        assert self._neighbors is not None
        metadata = metadata or self._default_metadata()
        if metadata.model_name != self.model_name:
            raise ValueError("artifact metadata model does not match recommender")
        if metadata.artifact_version != ARTIFACT_VERSION:
            raise ValueError("unsupported artifact version")
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "metadata": metadata.as_dict(),
            "model": {
                "configuration": self.config.as_dict(),
                "item_count": self.item_count,
                "neighbors": _matrix_payload(self._neighbors),
                "ratings": _matrix_payload(self._entity_matrix),
                "user_count": self.user_count,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.metadata = metadata
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read recommender artifact {path}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError("recommender artifact must contain an object")
        if value.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported artifact version")
        metadata_value = value.get("metadata")
        model_value = value.get("model")
        if not isinstance(metadata_value, Mapping) or not isinstance(model_value, Mapping):
            raise ValueError("recommender artifact is incomplete")
        metadata = ArtifactMetadata.from_dict(cast(Mapping[str, object], metadata_value))
        if metadata.model_name != cls.model_name:
            raise ValueError(f"artifact does not contain a {cls.model_name} recommender")
        configuration = model_value.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("artifact neighborhood configuration is missing")
        user_count = _positive_integer(model_value.get("user_count"), "user_count")
        item_count = _positive_integer(model_value.get("item_count"), "item_count")
        model = cls(
            user_count,
            item_count,
            n_neighbors=_positive_integer(configuration.get("n_neighbors"), "n_neighbors"),
            similarity=cast(str, configuration.get("similarity")),
            shrinkage=cast(float, configuration.get("shrinkage")),
            min_support=_positive_integer(configuration.get("min_support"), "min_support"),
            aggregation=cast(str, configuration.get("aggregation", "weighted_mean")),
        )
        entity_matrix = _matrix_from_payload(model_value.get("ratings"), "ratings")
        neighbors = _matrix_from_payload(model_value.get("neighbors"), "neighbors")
        expected_entity_shape = (
            (user_count, item_count) if cls.orientation == "user" else (item_count, user_count)
        )
        if entity_matrix.shape != expected_entity_shape:
            raise ValueError("artifact ratings shape does not match recommender orientation")
        if neighbors.shape != (expected_entity_shape[0], expected_entity_shape[0]):
            raise ValueError("artifact neighbors shape does not match recommender")
        presence = entity_matrix.copy()
        presence.data.fill(1.0)
        model._entity_matrix = entity_matrix
        model._presence_matrix = presence
        model._neighbors = neighbors
        model.metadata = metadata
        model._fitted = True
        return model


class UserKNNRecommender(_NeighborhoodRecommender):
    """Score candidates from cosine-similar users and their training ratings."""

    model_name = "user_knn"
    orientation = "user"


class ItemKNNRecommender(_NeighborhoodRecommender):
    """Score candidates from cosine-similar items and a user's training ratings."""

    model_name = "item_knn"
    orientation = "item"
