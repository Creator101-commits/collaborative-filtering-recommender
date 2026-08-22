"""Idempotent acquisition of the MovieLens 1M source dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from recommender.data.source import count_records, parse_movie, parse_rating, parse_user

DEFAULT_RAW_DIR = Path("data/raw/movielens-1m")
MANIFEST_NAME = "manifest.json"
SOURCE_FILES = ("README", "movies.dat", "ratings.dat", "users.dat")
PARSERS = {
    "movies.dat": parse_movie,
    "ratings.dat": parse_rating,
    "users.dat": parse_user,
}


class DatasetError(RuntimeError):
    """Raised when dataset acquisition or validation fails."""


@dataclass(frozen=True, slots=True)
class DatasetSource:
    url: str
    archive_size: int
    sha256: str
    md5: str
    record_counts: Mapping[str, int]


OFFICIAL_SOURCE = DatasetSource(
    url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    archive_size=5_917_549,
    sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
    md5="c4d9eecfca2ab87c1945afe126590906",
    record_counts={"movies.dat": 3_883, "ratings.dat": 1_000_209, "users.dat": 6_040},
)


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "recommender/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            with destination.open("wb") as archive:
                shutil.copyfileobj(response, archive)
    except OSError as error:
        raise DatasetError(f"failed to download {url}: {error}") from error


def _verify_archive(path: Path, source: DatasetSource) -> None:
    if path.stat().st_size != source.archive_size:
        raise DatasetError("downloaded archive has an unexpected size")
    if _digest(path, "sha256") != source.sha256:
        raise DatasetError("downloaded archive failed SHA-256 verification")
    if _digest(path, "md5") != source.md5:
        raise DatasetError("downloaded archive failed the publisher's MD5 verification")


def _extract_archive(archive: Path, destination: Path) -> None:
    expected = {f"ml-1m/{name}" for name in SOURCE_FILES}
    try:
        with zipfile.ZipFile(archive) as source:
            members = [member for member in source.infolist() if not member.is_dir()]
            if len(members) != len(expected) or {member.filename for member in members} != expected:
                raise DatasetError("archive does not contain the expected MovieLens 1M files")
            destination.mkdir()
            for member in members:
                target = destination / Path(member.filename).name
                with source.open(member) as input_file, target.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
    except zipfile.BadZipFile as error:
        raise DatasetError("downloaded file is not a valid ZIP archive") from error


def _validate_files(directory: Path, expected_counts: Mapping[str, int]) -> dict[str, int]:
    missing = [name for name in SOURCE_FILES if not (directory / name).is_file()]
    if missing:
        raise DatasetError(f"missing source files: {', '.join(missing)}")
    if not (directory / "README").stat().st_size:
        raise DatasetError("README is empty")

    counts: dict[str, int] = {}
    for name, parser in PARSERS.items():
        try:
            count = count_records(directory / name, parser)
        except (OSError, ValueError) as error:
            raise DatasetError(str(error)) from error
        if count != expected_counts[name]:
            raise DatasetError(f"{name} has {count} records; expected {expected_counts[name]}")
        counts[name] = count
    return counts


def _file_metadata(directory: Path, counts: Mapping[str, int]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for name in SOURCE_FILES:
        path = directory / name
        entry: dict[str, object] = {
            "sha256": _digest(path, "sha256"),
            "size_bytes": path.stat().st_size,
        }
        if name in counts:
            entry["records"] = counts[name]
        metadata[name] = entry
    return metadata


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"cannot read existing manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise DatasetError("existing manifest must contain a JSON object")
    return cast(dict[str, object], manifest)


def _validate_existing(directory: Path, source: DatasetSource) -> None:
    manifest = _load_manifest(directory / MANIFEST_NAME)
    if manifest.get("dataset") != "MovieLens 1M" or manifest.get("manifest_version") != 1:
        raise DatasetError("existing manifest has an unsupported format")
    source_entry = manifest.get("source")
    if not isinstance(source_entry, dict):
        raise DatasetError("existing manifest has no source metadata")
    expected_source = {
        "archive_size_bytes": source.archive_size,
        "md5": source.md5,
        "sha256": source.sha256,
        "url": source.url,
    }
    if any(source_entry.get(key) != value for key, value in expected_source.items()):
        raise DatasetError("existing manifest describes a different source archive")
    counts = _validate_files(directory, source.record_counts)
    if manifest.get("files") != _file_metadata(directory, counts):
        raise DatasetError("existing source files do not match the manifest")


def acquire_dataset(
    output_dir: Path = DEFAULT_RAW_DIR,
    *,
    source: DatasetSource = OFFICIAL_SOURCE,
) -> Path:
    """Download and validate MovieLens 1M, or reuse an identical valid copy."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        try:
            _validate_existing(output_dir, source)
        except DatasetError as error:
            raise DatasetError(
                f"{output_dir} exists but is not a valid reusable dataset: {error}"
            ) from error
        return output_dir / MANIFEST_NAME

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".movielens-1m-", dir=output_dir.parent) as temp:
        temp_dir = Path(temp)
        archive = temp_dir / "ml-1m.zip"
        staging = temp_dir / "dataset"
        _download(source.url, archive)
        _verify_archive(archive, source)
        _extract_archive(archive, staging)
        counts = _validate_files(staging, source.record_counts)
        manifest = {
            "dataset": "MovieLens 1M",
            "manifest_version": 1,
            "source": {
                "acquired_at_utc": datetime.now(UTC).isoformat(),
                "archive_size_bytes": archive.stat().st_size,
                "md5": source.md5,
                "sha256": source.sha256,
                "url": source.url,
            },
            "files": _file_metadata(staging, counts),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            staging.rename(output_dir)
        except FileExistsError as error:
            raise DatasetError(f"{output_dir} was created by another process") from error
    return output_dir / MANIFEST_NAME
