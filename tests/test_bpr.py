"""Tests for seeded Bayesian Personalized Ranking."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from recommender.models import (
    ArtifactMetadata,
    BPRConfig,
    BPRRecommender,
)
from recommender.models.bpr import _sample_negative_items

TRAIN = [
    {"user_idx": 0, "item_idx": 0, "rating": 5},
    {"user_idx": 0, "item_idx": 1, "rating": 1},
    {"user_idx": 1, "item_idx": 1, "rating": 5},
    {"user_idx": 1, "item_idx": 0, "rating": 1},
]
VALIDATION = [
    {"user_idx": 0, "item_idx": 2, "rating": 5},
    {"user_idx": 1, "item_idx": 3, "rating": 5},
]


def test_negative_sampling_is_seeded_and_excludes_observed_items() -> None:
    users = [0, 0, 1]
    blocked = {0: {0, 1, 2}, 1: {1, 3}}

    first = _sample_negative_items(users, blocked, 5, 2, random.Random(7))
    second = _sample_negative_items(users, blocked, 5, 2, random.Random(7))

    assert first == second
    sampled_users = [user_idx for user_idx in users for _ in range(2)]
    assert all(
        item_idx not in blocked[user_idx]
        for user_idx, item_idx in zip(sampled_users, first)
    )


def test_bpr_updates_parameters_and_overfits_tiny_fixture() -> None:
    model = BPRRecommender(
        2,
        4,
        latent_dim=3,
        learning_rate=0.05,
        regularization=0,
        epochs=50,
        batch_size=2,
    ).fit(TRAIN, seed=0)

    assert model.score(0, 0) > model.score(0, 2)
    assert model.score(1, 1) > model.score(1, 3)


def test_bpr_loss_has_gradients() -> None:
    model = BPRRecommender(1, 3, latent_dim=2)
    torch.manual_seed(0)
    network = model._make_network()
    loss = model._pairwise_loss(
        network,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
    )
    loss.backward()  # type: ignore[no-untyped-call]

    assert all(parameter.grad is not None for parameter in network.parameters())


def test_bpr_is_seeded_and_validation_checkpoint_survives_save_load(tmp_path: Path) -> None:
    first = BPRRecommender(
        2,
        5,
        latent_dim=3,
        learning_rate=0.05,
        regularization=0,
        epochs=10,
        batch_size=2,
    ).fit(TRAIN, seed=3, validation_interactions=VALIDATION)
    second = BPRRecommender(
        2,
        5,
        latent_dim=3,
        learning_rate=0.05,
        regularization=0,
        epochs=10,
        batch_size=2,
    ).fit(TRAIN, seed=3, validation_interactions=VALIDATION)

    assert first.best_epoch is not None
    assert first.validation_loss is not None
    assert first.score(0, 4) == pytest.approx(second.score(0, 4))

    metadata = ArtifactMetadata(
        model_name="bpr",
        data_manifest_sha256="manifest-hash",
        configuration={"model": "bpr"},
        seed=3,
        git_revision="revision",
        run_id="run",
    )
    path = first.save(tmp_path / "bpr.pt", metadata=metadata)
    restored = BPRRecommender.load(path)

    assert restored.metadata == metadata
    assert restored.best_epoch == first.best_epoch
    assert restored.validation_loss == pytest.approx(first.validation_loss)
    assert restored.score(0, 4) == pytest.approx(first.score(0, 4))
    assert restored.recommend(0, [1, 2, 3, 4], cutoff=2) == first.recommend(
        0, [1, 2, 3, 4], cutoff=2
    )


def test_bpr_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="negative_samples"):
        BPRConfig(negative_samples=0)
    with pytest.raises(ValueError, match="positive_rating_threshold"):
        BPRConfig(positive_rating_threshold=6)
