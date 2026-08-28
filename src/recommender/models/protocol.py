"""Protocol implemented by every recommender in the benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self

from recommender.models.artifacts import ArtifactMetadata

type Interaction = Mapping[str, Any]


class RecommenderProtocol(Protocol):
    """Common fit, scoring, recommendation, and artifact interface."""

    model_name: ClassVar[str]

    def fit(self, interactions: Iterable[Interaction], *, seed: int = 0) -> Self:
        """Fit the recommender using only the supplied interactions."""
        ...

    def score(self, user_idx: int, item_idx: int) -> float:
        """Return a score for one user-item pair."""
        ...

    def recommend(
        self, user_idx: int, candidates: Iterable[int], cutoff: int = 10
    ) -> Sequence[int]:
        """Rank the supplied candidate items for a user."""
        ...

    def save(self, path: Path, *, metadata: ArtifactMetadata | None = None) -> Path:
        """Serialize the fitted recommender and its metadata."""
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a serialized recommender artifact."""
        ...


class ValidationAwareRecommenderProtocol(Protocol):
    """Optional fit interface for models that select checkpoints on validation data."""

    def fit(
        self,
        interactions: Iterable[Interaction],
        *,
        seed: int = 0,
        validation_interactions: Iterable[Interaction] | None = None,
    ) -> Self:
        ...
