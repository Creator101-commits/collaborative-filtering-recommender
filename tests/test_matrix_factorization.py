"""Tests for seeded biased matrix factorization."""

from __future__ import annotations

from pathlib import Path

import pytest

from recommender.models import (
    ArtifactMetadata,
    BiasedMatrixFactorizationRecommender,
    MatrixFactorizationConfig,
)

ROWS = [
    {"user_idx": 0, "item_idx": 0, "rating": 5},
    {"user_idx": 0, "item_idx": 1, "rating": 1},
    {"user_idx": 1, "item_idx": 0, "rating": 1},
    {"user_idx": 1, "item_idx": 1, "rating": 5},
]


def test_matrix_factorization_is_seeded() -> None:
    first = BiasedMatrixFactorizationRecommender(
        2, 3, latent_dim=4, learning_rate=0.05, epochs=40, batch_size=2
    ).fit(ROWS, seed=7)
    second = BiasedMatrixFactorizationRecommender(
        2, 3, latent_dim=4, learning_rate=0.05, epochs=40, batch_size=2
    ).fit(ROWS, seed=7)

    assert first.recommend(0, [1, 2], cutoff=2) == second.recommend(0, [1, 2], cutoff=2)
    assert first.score(0, 0) == pytest.approx(second.score(0, 0))


def test_matrix_factorization_can_overfit_tiny_fixture() -> None:
    model = BiasedMatrixFactorizationRecommender(
        2,
        3,
        latent_dim=3,
        learning_rate=0.05,
        regularization=0.0,
        epochs=300,
        batch_size=4,
    ).fit(ROWS, seed=0)

    assert model.score(0, 0) > model.score(0, 1)
    assert model.score(1, 1) > model.score(1, 0)


def test_validation_checkpoint_and_save_load_parity(tmp_path: Path) -> None:
    model = BiasedMatrixFactorizationRecommender(
        2, 3, latent_dim=2, learning_rate=0.02, epochs=10, batch_size=2
    ).fit(
        ROWS,
        seed=3,
        validation_interactions=[{"user_idx": 0, "item_idx": 2, "rating": 3}],
    )
    metadata = ArtifactMetadata(
        model_name=model.model_name,
        data_manifest_sha256="manifest-hash",
        configuration={"model": model.model_name},
        seed=3,
        git_revision="revision",
    )

    path = model.save(tmp_path / "model.pt", metadata=metadata)
    restored = BiasedMatrixFactorizationRecommender.load(path)

    assert model.best_epoch is not None
    assert model.validation_loss is not None
    assert restored.metadata == metadata
    assert restored.score(0, 2) == pytest.approx(model.score(0, 2))
    assert restored.recommend(0, [1, 2], cutoff=2) == model.recommend(0, [1, 2], cutoff=2)


def test_matrix_factorization_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        MatrixFactorizationConfig(latent_dim=0)
    with pytest.raises(ValueError, match="learning_rate"):
        MatrixFactorizationConfig(learning_rate=0)
    with pytest.raises(ValueError, match="epochs"):
        MatrixFactorizationConfig(epochs=0)
