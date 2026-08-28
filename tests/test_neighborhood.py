"""Tests for sparse user- and item-neighborhood recommenders."""

from pathlib import Path

import pytest

from recommender.models import (
    ArtifactMetadata,
    ItemKNNRecommender,
    NeighborhoodConfig,
    UserKNNRecommender,
)

ROWS = [
    {"user_idx": 0, "item_idx": 0, "rating": 5},
    {"user_idx": 0, "item_idx": 1, "rating": 4},
    {"user_idx": 1, "item_idx": 0, "rating": 5},
    {"user_idx": 1, "item_idx": 2, "rating": 3},
    {"user_idx": 2, "item_idx": 1, "rating": 4},
    {"user_idx": 2, "item_idx": 2, "rating": 1},
]


@pytest.mark.parametrize("model_cls", [UserKNNRecommender, ItemKNNRecommender])
def test_knn_selects_deterministic_neighbors_and_scores(
    model_cls: type[UserKNNRecommender] | type[ItemKNNRecommender],
) -> None:
    model = model_cls(3, 4, n_neighbors=1, shrinkage=0, min_support=1).fit(ROWS)

    assert model.neighbor_indices[0] == (
        (1,) if model_cls is UserKNNRecommender else (2,)
    )
    assert model.recommend(0, [0, 2, 3], cutoff=2) == [2, 3]
    assert model.score(0, 2) > 0


@pytest.mark.parametrize("model_cls", [UserKNNRecommender, ItemKNNRecommender])
def test_knn_minimum_support_can_reject_neighbors(
    model_cls: type[UserKNNRecommender] | type[ItemKNNRecommender],
) -> None:
    model = model_cls(3, 4, n_neighbors=2, shrinkage=0, min_support=2).fit(ROWS)

    assert model.neighbor_indices[0] == ()
    assert model.recommend(0, [2, 3], cutoff=2) == [2, 3]


def test_neighborhood_configuration_rejects_unjustified_settings() -> None:
    with pytest.raises(ValueError, match="cosine or adjusted_cosine"):
        NeighborhoodConfig(similarity="pearson")
    with pytest.raises(ValueError, match="shrinkage"):
        NeighborhoodConfig(shrinkage=-1)
    with pytest.raises(ValueError, match="min_support"):
        NeighborhoodConfig(min_support=0)


def test_adjusted_cosine_is_supported() -> None:
    model = UserKNNRecommender(
        3, 4, n_neighbors=1, similarity="adjusted_cosine", shrinkage=0, min_support=1
    ).fit(ROWS)

    assert model.config.similarity == "adjusted_cosine"
    assert model.neighbor_indices[0] == (1,)


@pytest.mark.parametrize("model_cls", [UserKNNRecommender, ItemKNNRecommender])
def test_knn_save_load_preserves_recommendations(
    model_cls: type[UserKNNRecommender] | type[ItemKNNRecommender], tmp_path: Path
) -> None:
    model = model_cls(3, 4, n_neighbors=2, shrinkage=0, min_support=1).fit(ROWS)
    metadata = ArtifactMetadata(
        model_name=model.model_name,
        data_manifest_sha256="manifest-hash",
        configuration={"model": model.model_name},
        seed=3,
        git_revision="revision",
        run_id="run",
    )

    path = model.save(tmp_path / f"{model.model_name}.json", metadata=metadata)
    restored = model_cls.load(path)

    assert restored.metadata == metadata
    assert restored.neighbor_indices == model.neighbor_indices
    assert restored.recommend(0, [2, 3], cutoff=2) == model.recommend(0, [2, 3], cutoff=2)
    assert restored.score(0, 2) == pytest.approx(model.score(0, 2))
