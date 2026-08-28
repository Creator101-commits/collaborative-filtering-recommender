"""Seeded biased matrix factorization with validation checkpointing."""

from __future__ import annotations

import math
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import torch
from torch import nn

from recommender.models.artifacts import ARTIFACT_VERSION, ArtifactMetadata
from recommender.models.protocol import Interaction


@dataclass(frozen=True, slots=True)
class MatrixFactorizationConfig:
    """Training settings for the squared-error biased factorization model."""

    latent_dim: int = 32
    learning_rate: float = 0.01
    regularization: float = 0.0001
    epochs: int = 20
    batch_size: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.latent_dim, int) or isinstance(self.latent_dim, bool):
            raise ValueError("latent_dim must be an integer")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("regularization", self.regularization),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool):
            raise ValueError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
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


def _rows_to_tensors(
    interactions: Iterable[Interaction], user_count: int, item_count: int, label: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, set[int]]]:
    users: list[int] = []
    items: list[int] = []
    ratings: list[float] = []
    seen_pairs: set[tuple[int, int]] = set()
    seen_by_user: dict[int, set[int]] = {}
    for row in interactions:
        user_idx = _integer_field(row, "user_idx")
        item_idx = _integer_field(row, "item_idx")
        if not 0 <= user_idx < user_count:
            raise ValueError(f"{label} user_idx is outside the user catalog")
        if not 0 <= item_idx < item_count:
            raise ValueError(f"{label} item_idx is outside the item catalog")
        if (user_idx, item_idx) in seen_pairs:
            raise ValueError(f"{label} interactions contain duplicate user/item pairs")
        seen_pairs.add((user_idx, item_idx))
        try:
            rating = float(row["rating"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} interaction has an invalid rating") from error
        if not math.isfinite(rating):
            raise ValueError(f"{label} rating must be finite")
        users.append(user_idx)
        items.append(item_idx)
        ratings.append(rating)
        seen_by_user.setdefault(user_idx, set()).add(item_idx)
    if not ratings:
        raise ValueError(f"{label} interactions must not be empty")
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(items, dtype=torch.long),
        torch.tensor(ratings, dtype=torch.float32),
        seen_by_user,
    )


class BiasedMatrixFactorizationRecommender:
    """Predict ratings using user/item factors, biases, and a global bias."""

    model_name: ClassVar[str] = "matrix_factorization"

    def __init__(
        self,
        user_count: int,
        item_count: int,
        *,
        latent_dim: int = 32,
        learning_rate: float = 0.01,
        regularization: float = 0.0001,
        epochs: int = 20,
        batch_size: int = 1024,
    ) -> None:
        if not isinstance(user_count, int) or isinstance(user_count, bool) or user_count <= 0:
            raise ValueError("user_count must be a positive integer")
        if not isinstance(item_count, int) or isinstance(item_count, bool) or item_count <= 0:
            raise ValueError("item_count must be a positive integer")
        self.user_count = user_count
        self.item_count = item_count
        self.config = MatrixFactorizationConfig(
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            regularization=regularization,
            epochs=epochs,
            batch_size=batch_size,
        )
        self._network: _BiasedModel | None = None
        self._seen_by_user: dict[int, set[int]] = {}
        self.metadata: ArtifactMetadata | None = None
        self.best_epoch: int | None = None
        self.validation_loss: float | None = None
        self._fitted = False

    def _check_fitted(self) -> _BiasedModel:
        if not self._fitted or self._network is None:
            raise RuntimeError("recommender must be fitted before use")
        return self._network

    def _make_network(self, global_mean: float) -> _BiasedModel:
        network = _BiasedModel(self.user_count, self.item_count, self.config.latent_dim)
        nn.init.normal_(network.user_factors.weight, mean=0.0, std=0.05)
        nn.init.normal_(network.item_factors.weight, mean=0.0, std=0.05)
        nn.init.zeros_(network.user_bias.weight)
        nn.init.zeros_(network.item_bias.weight)
        network.global_bias.data.fill_(global_mean)
        return network

    @staticmethod
    def _predict(
        network: _BiasedModel, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        model = network
        return cast(
            torch.Tensor,
            model.global_bias
            + model.user_bias(users).squeeze(-1)
            + model.item_bias(items).squeeze(-1)
            + (model.user_factors(users) * model.item_factors(items)).sum(dim=1),
        )

    def fit(
        self,
        interactions: Iterable[Interaction],
        *,
        seed: int = 0,
        validation_interactions: Iterable[Interaction] | None = None,
    ) -> Self:
        train_users, train_items, train_ratings, seen_by_user = _rows_to_tensors(
            interactions, self.user_count, self.item_count, "training"
        )
        validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if validation_interactions is not None:
            validation_users, validation_items, validation_ratings, _ = _rows_to_tensors(
                validation_interactions, self.user_count, self.item_count, "validation"
            )
            validation = (validation_users, validation_items, validation_ratings)

        torch.manual_seed(seed)
        network = self._make_network(float(train_ratings.mean()))
        optimizer = torch.optim.SGD(network.parameters(), lr=self.config.learning_rate)
        generator = torch.Generator().manual_seed(seed)
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        best_epoch: int | None = None

        for epoch in range(self.config.epochs):
            order = torch.randperm(train_users.numel(), generator=generator)
            network.train()
            for start in range(0, train_users.numel(), self.config.batch_size):
                batch = order[start : start + self.config.batch_size]
                predictions = self._predict(network, train_users[batch], train_items[batch])
                error = predictions - train_ratings[batch]
                factors = network.user_factors(train_users[batch]), network.item_factors(
                    train_items[batch]
                )
                penalty = sum(parameter.square().mean() for parameter in factors)
                loss = error.square().mean() + self.config.regularization * penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if validation is None:
                best_epoch = epoch + 1
                continue
            network.eval()
            with torch.no_grad():
                validation_predictions = self._predict(network, validation[0], validation[1])
                current_loss = float((validation_predictions - validation[2]).square().mean())
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch + 1
                best_state = {
                    name: parameter.detach().clone()
                    for name, parameter in network.state_dict().items()
                }

        if validation is not None and best_state is not None:
            network.load_state_dict(best_state)
            self.validation_loss = best_loss
        else:
            self.validation_loss = None
        self._network = network
        self._seen_by_user = seen_by_user
        self.best_epoch = best_epoch
        self.metadata = None
        self._fitted = True
        return self

    def _score_items(self, user_idx: int, item_indices: Iterable[int]) -> list[float]:
        network = self._check_fitted()
        if not 0 <= user_idx < self.user_count:
            raise ValueError("user_idx is outside the user catalog")
        items = list(item_indices)
        if any(not 0 <= item_idx < self.item_count for item_idx in items):
            raise ValueError("item_idx is outside the item catalog")
        if not items:
            return []
        users = torch.full((len(items),), user_idx, dtype=torch.long)
        item_tensor = torch.tensor(items, dtype=torch.long)
        network.eval()
        with torch.no_grad():
            return [float(value) for value in self._predict(network, users, item_tensor)]

    def score(self, user_idx: int, item_idx: int) -> float:
        return self._score_items(user_idx, [item_idx])[0]

    def recommend(self, user_idx: int, candidates: Iterable[int], cutoff: int = 10) -> list[int]:
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        candidate_set = set(candidates)
        if any(
            not isinstance(item_idx, int) or isinstance(item_idx, bool)
            for item_idx in candidate_set
        ):
            raise ValueError("candidate item indices must be integers")
        if any(not 0 <= item_idx < self.item_count for item_idx in candidate_set):
            raise ValueError("item_idx is outside the item catalog")
        candidate_set.difference_update(self._seen_by_user.get(user_idx, set()))
        scores = dict(zip(candidate_set, self._score_items(user_idx, candidate_set)))
        return sorted(candidate_set, key=lambda item_idx: (-scores[item_idx], item_idx))[:cutoff]

    def save(self, path: Path, *, metadata: ArtifactMetadata | None = None) -> Path:
        network = self._check_fitted()
        metadata = metadata or ArtifactMetadata(
            model_name=self.model_name,
            data_manifest_sha256="",
            configuration={"matrix_factorization": self.config.as_dict()},
            seed=0,
            git_revision="unknown",
        )
        if metadata.model_name != self.model_name:
            raise ValueError("artifact metadata model does not match recommender")
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "metadata": metadata.as_dict(),
            "model": {
                "config": self.config.as_dict(),
                "item_count": self.item_count,
                "network": network.state_dict(),
                "seen_by_user": {
                    str(user_idx): sorted(items) for user_idx, items in self._seen_by_user.items()
                },
                "user_count": self.user_count,
                "best_epoch": self.best_epoch,
                "validation_loss": self.validation_loss,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self.metadata = metadata
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            raise ValueError(f"cannot read recommender artifact {path}: {error}") from error
        if not isinstance(value, Mapping) or value.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported or invalid recommender artifact")
        metadata_value = value.get("metadata")
        model_value = value.get("model")
        if not isinstance(metadata_value, Mapping) or not isinstance(model_value, Mapping):
            raise ValueError("recommender artifact is incomplete")
        metadata = ArtifactMetadata.from_dict(cast(Mapping[str, object], metadata_value))
        if metadata.model_name != cls.model_name:
            raise ValueError("artifact does not contain a matrix factorization recommender")
        config_value = model_value.get("config")
        if not isinstance(config_value, Mapping):
            raise ValueError("artifact matrix factorization configuration is missing")
        config = MatrixFactorizationConfig(
            latent_dim=cast(int, config_value.get("latent_dim")),
            learning_rate=cast(float, config_value.get("learning_rate")),
            regularization=cast(float, config_value.get("regularization")),
            epochs=cast(int, config_value.get("epochs")),
            batch_size=cast(int, config_value.get("batch_size")),
        )
        user_count = model_value.get("user_count")
        item_count = model_value.get("item_count")
        if not isinstance(user_count, int) or isinstance(user_count, bool):
            raise ValueError("artifact user_count must be an integer")
        if not isinstance(item_count, int) or isinstance(item_count, bool):
            raise ValueError("artifact item_count must be an integer")
        model = cls(
            user_count,
            item_count,
            latent_dim=config.latent_dim,
            learning_rate=config.learning_rate,
            regularization=config.regularization,
            epochs=config.epochs,
            batch_size=config.batch_size,
        )
        network = model._make_network(0.0)
        state_dict = model_value.get("network")
        if not isinstance(state_dict, Mapping):
            raise ValueError("artifact network state is missing")
        try:
            network.load_state_dict(state_dict)
        except (RuntimeError, TypeError) as error:
            raise ValueError("artifact network state is invalid") from error
        seen_value = model_value.get("seen_by_user", {})
        if not isinstance(seen_value, Mapping):
            raise ValueError("artifact seen-item map is invalid")
        seen_by_user: dict[int, set[int]] = {}
        for user_key, items in seen_value.items():
            try:
                user_idx = int(user_key)
            except (TypeError, ValueError) as error:
                raise ValueError("artifact seen-item map has an invalid user") from error
            if not isinstance(items, list) or any(
                not isinstance(item, int) or isinstance(item, bool) for item in items
            ):
                raise ValueError("artifact seen-item map has invalid items")
            if not 0 <= user_idx < model.user_count or any(
                not 0 <= item < model.item_count for item in items
            ):
                raise ValueError("artifact seen-item map is outside the catalog")
            seen_by_user[user_idx] = set(items)
        model._network = network
        model._seen_by_user = seen_by_user
        model.best_epoch = cast(int | None, model_value.get("best_epoch"))
        model.validation_loss = cast(float | None, model_value.get("validation_loss"))
        model.metadata = metadata
        model._fitted = True
        return model


class _BiasedModel(nn.Module):
    def __init__(self, user_count: int, item_count: int, latent_dim: int) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(user_count, latent_dim)
        self.item_factors = nn.Embedding(item_count, latent_dim)
        self.user_bias = nn.Embedding(user_count, 1)
        self.item_bias = nn.Embedding(item_count, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))


MatrixFactorizationRecommender = BiasedMatrixFactorizationRecommender
