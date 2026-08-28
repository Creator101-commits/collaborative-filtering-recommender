"""Tests for the popularity baseline and artifact parity."""

from pathlib import Path

import pytest

from recommender.models import ArtifactMetadata, PopularityRecommender


@pytest.fixture
def fitted_model() -> PopularityRecommender:
    return PopularityRecommender(4).fit(
        [
            {"item_idx": 0, "rating": 5},
            {"item_idx": 1, "rating": 4},
            {"item_idx": 1, "rating": 5},
            {"item_idx": 2, "rating": 3},
        ]
    )


def test_popularity_ranks_positive_training_counts(fitted_model: PopularityRecommender) -> None:
    assert fitted_model.item_popularity == {0: 1, 1: 2, 2: 0, 3: 0}
    assert fitted_model.recommend(0, [0, 1, 2, 3], cutoff=3) == [1, 0, 2]
    assert fitted_model.score(0, 1) == 2.0


def test_popularity_save_load_preserves_recommendations(
    fitted_model: PopularityRecommender, tmp_path: Path
) -> None:
    metadata = ArtifactMetadata(
        model_name="popularity",
        data_manifest_sha256="manifest-hash",
        configuration={"cutoff": 10},
        seed=7,
        git_revision="revision",
        run_id="run",
    )
    path = fitted_model.save(tmp_path / "model.json", metadata=metadata)

    restored = PopularityRecommender.load(path)

    assert restored.recommend(0, [0, 1, 2, 3], cutoff=3) == [1, 0, 2]
    assert restored.metadata == metadata


def test_popularity_rejects_invalid_cutoff(fitted_model: PopularityRecommender) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        fitted_model.recommend(0, [0], cutoff=0)
