"""Tests for sparse user- and item-neighborhood recommenders."""

import json
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
    with pytest.raises(ValueError, match="aggregation"):
        NeighborhoodConfig(aggregation="mean")


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


# Item 0 is reached only through one similar neighbor's high rating while
# item 1 is reached through several neighbors' slightly lower ratings.
USER_KNN_SINGLE_VS_MANY_ROWS = [
    {"user_idx": 0, "item_idx": 2, "rating": 5},
    {"user_idx": 0, "item_idx": 3, "rating": 5},
    {"user_idx": 0, "item_idx": 4, "rating": 5},
    {"user_idx": 1, "item_idx": 0, "rating": 5},
    {"user_idx": 1, "item_idx": 2, "rating": 5},
    {"user_idx": 1, "item_idx": 3, "rating": 5},
    {"user_idx": 1, "item_idx": 4, "rating": 5},
    {"user_idx": 2, "item_idx": 1, "rating": 4},
    {"user_idx": 2, "item_idx": 2, "rating": 5},
    {"user_idx": 2, "item_idx": 3, "rating": 5},
    {"user_idx": 2, "item_idx": 4, "rating": 5},
    {"user_idx": 3, "item_idx": 1, "rating": 4},
    {"user_idx": 3, "item_idx": 2, "rating": 5},
    {"user_idx": 3, "item_idx": 3, "rating": 5},
    {"user_idx": 3, "item_idx": 4, "rating": 5},
    {"user_idx": 4, "item_idx": 1, "rating": 4},
    {"user_idx": 4, "item_idx": 2, "rating": 5},
    {"user_idx": 4, "item_idx": 3, "rating": 5},
    {"user_idx": 4, "item_idx": 4, "rating": 5},
]

ITEM_KNN_SINGLE_VS_MANY_ROWS = [
    {"user_idx": 0, "item_idx": 2, "rating": 5},
    {"user_idx": 0, "item_idx": 3, "rating": 4},
    {"user_idx": 0, "item_idx": 4, "rating": 4},
    {"user_idx": 1, "item_idx": 0, "rating": 5},
    {"user_idx": 1, "item_idx": 2, "rating": 5},
    {"user_idx": 2, "item_idx": 1, "rating": 4},
    {"user_idx": 2, "item_idx": 2, "rating": 4},
    {"user_idx": 2, "item_idx": 3, "rating": 4},
    {"user_idx": 2, "item_idx": 4, "rating": 4},
]


@pytest.mark.parametrize(
    ("model_cls", "rows", "user_count", "item_count"),
    [
        (UserKNNRecommender, USER_KNN_SINGLE_VS_MANY_ROWS, 5, 5),
        (ItemKNNRecommender, ITEM_KNN_SINGLE_VS_MANY_ROWS, 3, 5),
    ],
)
def test_knn_weighted_sum_prefers_items_supported_by_many_neighbors(
    model_cls: type[UserKNNRecommender] | type[ItemKNNRecommender],
    rows: list[dict[str, int]],
    user_count: int,
    item_count: int,
) -> None:
    mean_model = model_cls(
        user_count,
        item_count,
        n_neighbors=10,
        shrinkage=0,
        min_support=1,
        aggregation="weighted_mean",
    ).fit(rows)
    sum_model = model_cls(
        user_count,
        item_count,
        n_neighbors=10,
        shrinkage=0,
        min_support=1,
        aggregation="weighted_sum",
    ).fit(rows)

    assert mean_model.recommend(0, [0, 1], cutoff=2) == [0, 1]
    assert sum_model.recommend(0, [0, 1], cutoff=2) == [1, 0]


@pytest.mark.parametrize("model_cls", [UserKNNRecommender, ItemKNNRecommender])
def test_knn_save_load_round_trips_aggregation(
    model_cls: type[UserKNNRecommender] | type[ItemKNNRecommender], tmp_path: Path
) -> None:
    model = model_cls(
        3, 4, n_neighbors=2, shrinkage=0, min_support=1, aggregation="weighted_sum"
    ).fit(ROWS)
    path = model.save(tmp_path / f"{model.model_name}.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"]["configuration"]["aggregation"] == "weighted_sum"

    restored = model_cls.load(path)
    assert restored.config.aggregation == "weighted_sum"
    assert restored.recommend(0, [2, 3], cutoff=2) == model.recommend(
        0, [2, 3], cutoff=2
    )

    del payload["model"]["configuration"]["aggregation"]
    legacy_path = tmp_path / f"{model.model_name}-legacy.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = model_cls.load(legacy_path)
    assert legacy.config.aggregation == "weighted_mean"
