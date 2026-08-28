"""Metadata shared by serialized recommender artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self, cast

ARTIFACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Reproduction information stored alongside a model artifact."""

    model_name: str
    data_manifest_sha256: str
    configuration: Mapping[str, object]
    seed: int
    git_revision: str
    run_id: str | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    artifact_version: int = ARTIFACT_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata."""
        return {
            "artifact_version": self.artifact_version,
            "created_at_utc": self.created_at_utc,
            "data_manifest_sha256": self.data_manifest_sha256,
            "configuration": dict(self.configuration),
            "git_revision": self.git_revision,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        """Restore metadata and reject incomplete artifact records."""
        configuration = value.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("artifact metadata configuration must be an object")
        model_name = value.get("model_name")
        data_manifest_sha256 = value.get("data_manifest_sha256")
        git_revision = value.get("git_revision")
        created_at_utc = value.get("created_at_utc")
        seed = value.get("seed")
        artifact_version = value.get("artifact_version")
        run_id = value.get("run_id")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("artifact metadata has no model name")
        if not isinstance(data_manifest_sha256, str):
            raise ValueError("artifact metadata has no data manifest hash")
        if not isinstance(git_revision, str):
            raise ValueError("artifact metadata has no Git revision")
        if not isinstance(created_at_utc, str):
            raise ValueError("artifact metadata has no creation time")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("artifact metadata seed must be an integer")
        if artifact_version != ARTIFACT_VERSION:
            raise ValueError("unsupported artifact version")
        if run_id is not None and not isinstance(run_id, str):
            raise ValueError("artifact metadata run ID must be a string")
        return cls(
            artifact_version=artifact_version,
            created_at_utc=created_at_utc,
            data_manifest_sha256=data_manifest_sha256,
            configuration=cast(Mapping[str, object], configuration),
            git_revision=git_revision,
            model_name=model_name,
            run_id=run_id,
            seed=seed,
        )
