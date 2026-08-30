"""Frozen comparison protocol and pre-test input audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from recommender.evaluation import generate_candidate_catalog

PROTOCOL_VERSION = "phase-8-v1"
FINAL_MODEL_NAMES = (
    "popularity",
    "user_knn",
    "item_knn",
    "matrix_factorization",
    "bpr",
)
_REQUIRED_INPUTS = (
    "manifest.json",
    "movies.parquet",
    "users.parquet",
    "train.parquet",
    "validation.parquet",
)


class ProtocolError(RuntimeError):
    """Raised when the frozen comparison protocol cannot be applied."""


@dataclass(frozen=True, slots=True)
class ComparisonProtocol:
    """Decisions that must not change after final test results are opened."""

    version: str = PROTOCOL_VERSION
    metric_cutoffs: tuple[int, ...] = (5, 10, 20)
    primary_cutoff: int = 10
    learned_model_seeds: tuple[int, ...] = (0, 1, 2)
    bootstrap_replicates: int = 10_000
    confidence_level: float = 0.95
    history_quantiles: tuple[float, float] = (1 / 3, 2 / 3)
    item_popularity_quantiles: tuple[float, float] = (0.2, 0.8)
    novelty: str = "laplace_smoothed_self_information_bits"

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must not be empty")
        if (
            not self.metric_cutoffs
            or tuple(sorted(set(self.metric_cutoffs))) != self.metric_cutoffs
        ):
            raise ValueError("metric_cutoffs must be sorted and unique")
        if any(cutoff <= 0 for cutoff in self.metric_cutoffs):
            raise ValueError("metric_cutoffs must be positive")
        if self.primary_cutoff not in self.metric_cutoffs:
            raise ValueError("primary_cutoff must be one of metric_cutoffs")
        if not self.learned_model_seeds:
            raise ValueError("learned_model_seeds must not be empty")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        for name, quantiles in (
            ("history_quantiles", self.history_quantiles),
            ("item_popularity_quantiles", self.item_popularity_quantiles),
        ):
            if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
                raise ValueError(f"{name} must contain two increasing quantiles")
        if self.novelty != "laplace_smoothed_self_information_bits":
            raise ValueError("novelty must use the frozen self-information formulation")

    def as_dict(self) -> dict[str, object]:
        """Return the protocol as JSON-compatible data."""
        return {
            "bootstrap": {
                "confidence_level": self.confidence_level,
                "level": "user",
                "replicates": self.bootstrap_replicates,
                "interval": "percentile",
            },
            "cutoffs": list(self.metric_cutoffs),
            "compute_budget": {
                "knn_validation_configurations_per_family": 2,
                "learned_validation_configurations_per_model": 4,
                "popularity_validation_configurations": 1,
            },
            "history_slices": {
                "boundaries": list(self.history_quantiles),
                "definition": "training interaction-count quantiles",
            },
            "item_popularity_slices": {
                "boundaries": list(self.item_popularity_quantiles),
                "definition": "positive training interaction-count quantiles",
            },
            "learned_model_seeds": list(self.learned_model_seeds),
            "metrics": {
                "ranking": ["recall", "ndcg", "mrr", "hit_rate"],
                "beyond_accuracy": [
                    "catalog_coverage",
                    "mean_popularity",
                    "novelty",
                    "gini_exposure",
                ],
                "relevance": "binary rating-threshold holdouts",
            },
            "novelty": self.novelty,
            "primary_cutoff": self.primary_cutoff,
            "selection": "one configuration per model selected using validation only",
            "slices": [
                "history_length",
                "item_popularity",
                "cold_start_proxy",
                "recommendation_concentration",
            ],
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ProtocolAudit:
    """Evidence that the prepared inputs satisfy the frozen protocol."""

    data_manifest_sha256: str
    item_count: int
    user_count: int
    evaluated_users: int
    candidate_catalog_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = pq.read_table(path).to_pylist()
    except (OSError, pa.ArrowException) as error:
        raise ProtocolError(f"cannot read {path}: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ProtocolError(f"{path} has invalid rows")
    return [cast(dict[str, Any], row) for row in rows]


def _catalog_size(rows: list[dict[str, Any]], field: str, label: str) -> int:
    try:
        values = {int(row[field]) for row in rows}
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(f"{label} has invalid {field} values") from error
    if values != set(range(len(values))):
        raise ProtocolError(f"{label} {field} values are not a contiguous catalog")
    return len(values)


def audit_comparison_inputs(
    processed_dir: Path,
    model_configurations: Mapping[str, Mapping[str, object]],
    *,
    protocol: ComparisonProtocol | None = None,
) -> ProtocolAudit:
    """Audit model parity and validation inputs without reading test labels."""
    protocol = protocol or ComparisonProtocol()
    if set(model_configurations) != set(FINAL_MODEL_NAMES):
        raise ProtocolError("model configurations must contain every final model")
    if any(
        not isinstance(configuration, Mapping) for configuration in model_configurations.values()
    ):
        raise ProtocolError("model configurations must be mappings")

    processed_dir = processed_dir.resolve()
    manifest_path = processed_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("dataset") != "MovieLens 1M":
        raise ProtocolError("processed manifest is not MovieLens 1M")
    configuration = manifest.get("configuration")
    if (
        not isinstance(configuration, Mapping)
        or configuration.get("positive_rating_threshold") != 4
    ):
        raise ProtocolError("processed manifest does not use the frozen positive threshold")
    split_protocol = manifest.get("split_protocol")
    if not isinstance(split_protocol, Mapping):
        raise ProtocolError("processed manifest has no split protocol")
    if (
        split_protocol.get("candidate_catalog")
        != "all indexed movies except each user's training items"
    ):
        raise ProtocolError("processed manifest has an incompatible candidate policy")
    if split_protocol.get("training_events") != "all source interactions earlier than validation":
        raise ProtocolError("processed manifest has an incompatible training policy")
    for name in _REQUIRED_INPUTS:
        if not (processed_dir / name).is_file():
            raise ProtocolError(f"processed inputs are missing {name}")

    item_count = _catalog_size(_read_rows(processed_dir / "movies.parquet"), "item_idx", "movies")
    user_count = _catalog_size(_read_rows(processed_dir / "users.parquet"), "user_idx", "users")
    train_rows = _read_rows(processed_dir / "train.parquet")
    validation_rows = _read_rows(processed_dir / "validation.parquet")
    try:
        evaluated_users = sorted({int(row["user_idx"]) for row in validation_rows})
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("validation has invalid user indices") from error
    if not evaluated_users or any(user < 0 or user >= user_count for user in evaluated_users):
        raise ProtocolError("validation users are outside the user catalog")
    candidates = generate_candidate_catalog(item_count, train_rows, user_indices=evaluated_users)
    if any(not candidates[user] for user in evaluated_users):
        raise ProtocolError("an evaluated user has no eligible candidates")
    candidate_bytes = json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
    return ProtocolAudit(
        data_manifest_sha256=_sha256(manifest_path),
        item_count=item_count,
        user_count=user_count,
        evaluated_users=len(evaluated_users),
        candidate_catalog_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
    )


def write_experiment_card(
    path: Path,
    data_manifest_sha256: str,
    model_configurations: Mapping[str, Mapping[str, object]],
    *,
    git_revision: str,
    protocol: ComparisonProtocol | None = None,
) -> Path:
    """Write an immutable JSON experiment card before test evaluation."""
    protocol = protocol or ComparisonProtocol()
    if not git_revision:
        raise ValueError("git_revision must not be empty")
    if len(data_manifest_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in data_manifest_sha256
    ):
        raise ValueError("data_manifest_sha256 must be a lowercase SHA-256 digest")
    if set(model_configurations) != set(FINAL_MODEL_NAMES):
        raise ProtocolError("model configurations must contain every final model")
    card = {
        "card_version": 1,
        "git_revision": git_revision,
        "data_manifest_sha256": data_manifest_sha256,
        "models": {name: dict(model_configurations[name]) for name in FINAL_MODEL_NAMES},
        "protocol": protocol.as_dict(),
    }
    encoded = json.dumps(card, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == encoded:
                return path
        except OSError as error:
            raise ProtocolError(f"cannot read existing experiment card {path}: {error}") from error
        raise ProtocolError(f"experiment card already exists with different contents: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return path


def read_experiment_card(
    path: Path, *, protocol: ComparisonProtocol | None = None
) -> dict[str, object]:
    """Read and validate an immutable experiment card."""
    protocol = protocol or ComparisonProtocol()
    card = _read_json(path)
    if card.get("card_version") != 1:
        raise ProtocolError("experiment card has an unsupported version")
    if not isinstance(card.get("git_revision"), str) or not card["git_revision"]:
        raise ProtocolError("experiment card has no Git revision")
    if not isinstance(card.get("data_manifest_sha256"), str):
        raise ProtocolError("experiment card has no data-manifest hash")
    models = card.get("models")
    if not isinstance(models, Mapping) or set(models) != set(FINAL_MODEL_NAMES):
        raise ProtocolError("experiment card must contain every final model")
    if any(not isinstance(config, Mapping) for config in models.values()):
        raise ProtocolError("experiment card model configurations must be mappings")
    recorded_protocol = card.get("protocol")
    if recorded_protocol != protocol.as_dict():
        raise ProtocolError("experiment card does not match the frozen protocol")
    return card
