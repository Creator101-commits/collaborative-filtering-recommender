"""Deterministic non-personalized popularity recommender."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import ClassVar, Self, cast

from recommender.models.artifacts import ArtifactMetadata
from recommender.models.protocol import Interaction


class PopularityRecommender:
    """Rank items by positive training-interaction count.

    Ties are broken by the stable zero-based item index. Recommendations are
    restricted to the candidate sequence supplied by the evaluator.
    """

    model_name: ClassVar[str] = "popularity"

    def __init__(self, item_count: int, positive_rating_threshold: int = 4) -> None:
        if item_count <= 0:
            raise ValueError("item_count must be positive")
        if not 1 <= positive_rating_threshold <= 5:
            raise ValueError("positive_rating_threshold must be between 1 and 5")
        self.item_count = item_count
        self.positive_rating_threshold = positive_rating_threshold
        self.item_counts = [0] * item_count
        self.metadata: ArtifactMetadata | None = None
        self._fitted = False

    @property
    def item_popularity(self) -> dict[int, int]:
        """Return training popularity for every indexed item."""
        return dict(enumerate(self.item_counts))

    def fit(self, interactions: Iterable[Interaction], *, seed: int = 0) -> Self:
        """Count positive interactions without using user demographics or labels."""
        del seed
        self.item_counts = [0] * self.item_count
        for row in interactions:
            try:
                item_idx = int(row["item_idx"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("training interaction has no valid item_idx") from error
            if not 0 <= item_idx < self.item_count:
                raise ValueError(f"item_idx {item_idx} is outside the catalog")
            rating = row.get("rating", self.positive_rating_threshold)
            try:
                is_positive = int(rating) >= self.positive_rating_threshold
            except (TypeError, ValueError) as error:
                raise ValueError("training interaction has an invalid rating") from error
            if is_positive:
                self.item_counts[item_idx] += 1
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("recommender must be fitted before use")

    def score(self, user_idx: int, item_idx: int) -> float:
        """Return positive training count; user_idx is intentionally unused."""
        del user_idx
        self._check_fitted()
        if not 0 <= item_idx < self.item_count:
            raise ValueError(f"item_idx {item_idx} is outside the catalog")
        return float(self.item_counts[item_idx])

    def recommend(self, user_idx: int, candidates: Iterable[int], cutoff: int = 10) -> list[int]:
        """Return the highest-count candidate items in deterministic order."""
        del user_idx
        self._check_fitted()
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        candidate_set: set[int] = set()
        for item_idx in candidates:
            if not 0 <= item_idx < self.item_count:
                raise ValueError(f"item_idx {item_idx} is outside the catalog")
            candidate_set.add(item_idx)
        return sorted(candidate_set, key=lambda item: (-self.item_counts[item], item))[:cutoff]

    def save(self, path: Path, *, metadata: ArtifactMetadata | None = None) -> Path:
        """Write a versioned JSON model artifact."""
        self._check_fitted()
        if metadata is None:
            metadata = ArtifactMetadata(
                model_name=self.model_name,
                data_manifest_sha256="",
                configuration={
                    "positive_rating_threshold": self.positive_rating_threshold,
                },
                seed=0,
                git_revision="unknown",
            )
        if metadata.model_name != self.model_name:
            raise ValueError("artifact metadata model does not match recommender")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": metadata.artifact_version,
            "metadata": metadata.as_dict(),
            "model": {
                "item_count": self.item_count,
                "item_counts": self.item_counts,
                "positive_rating_threshold": self.positive_rating_threshold,
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.metadata = metadata
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load and validate a JSON popularity artifact."""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read recommender artifact {path}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError("recommender artifact must contain an object")
        metadata_value = value.get("metadata")
        model_value = value.get("model")
        if not isinstance(metadata_value, Mapping) or not isinstance(model_value, Mapping):
            raise ValueError("recommender artifact is incomplete")
        metadata = ArtifactMetadata.from_dict(cast(Mapping[str, object], metadata_value))
        if metadata.model_name != cls.model_name:
            raise ValueError("artifact does not contain a popularity recommender")
        item_count = model_value.get("item_count")
        threshold = model_value.get("positive_rating_threshold")
        counts = model_value.get("item_counts")
        if not isinstance(item_count, int) or isinstance(item_count, bool):
            raise ValueError("artifact item_count must be an integer")
        if not isinstance(threshold, int) or isinstance(threshold, bool):
            raise ValueError("artifact rating threshold must be an integer")
        if not isinstance(counts, list) or len(counts) != item_count:
            raise ValueError("artifact item counts do not match item_count")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in counts
        ):
            raise ValueError("artifact item counts must be non-negative integers")
        model = cls(item_count, threshold)
        model.item_counts = [int(count) for count in counts]
        model.metadata = metadata
        model._fitted = True
        return model
