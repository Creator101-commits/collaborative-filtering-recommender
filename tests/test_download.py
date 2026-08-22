"""Tests for idempotent MovieLens acquisition."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from recommender.data.download import DatasetError, DatasetSource, acquire_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ml-1m"
RECORD_COUNTS = {"movies.dat": 2, "ratings.dat": 2, "users.dat": 2}


def _source_archive(
    tmp_path: Path,
    *,
    replacements: dict[str, bytes] | None = None,
    extra_member: str | None = None,
) -> tuple[Path, DatasetSource]:
    archive = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for fixture in FIXTURE_DIR.iterdir():
            content = (replacements or {}).get(fixture.name, fixture.read_bytes())
            output.writestr(f"ml-1m/{fixture.name}", content)
        if extra_member:
            output.writestr(extra_member, b"unexpected")
    content = archive.read_bytes()
    source = DatasetSource(
        url=archive.as_uri(),
        archive_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        md5=hashlib.md5(content).hexdigest(),  # noqa: S324
        record_counts=RECORD_COUNTS,
    )
    return archive, source


def test_acquisition_is_idempotent(tmp_path: Path) -> None:
    _, source = _source_archive(tmp_path)
    output_dir = tmp_path / "raw"

    manifest_path = acquire_dataset(output_dir, source=source)
    first_manifest = manifest_path.read_bytes()

    assert acquire_dataset(output_dir, source=source) == manifest_path
    assert manifest_path.read_bytes() == first_manifest
    manifest = json.loads(first_manifest)
    assert manifest["source"]["sha256"] == source.sha256
    assert manifest["files"]["ratings.dat"]["records"] == 2


def test_invalid_hash_leaves_no_dataset(tmp_path: Path) -> None:
    _, source = _source_archive(tmp_path)
    invalid_source = DatasetSource(
        url=source.url,
        archive_size=source.archive_size,
        sha256="0" * 64,
        md5=source.md5,
        record_counts=source.record_counts,
    )
    output_dir = tmp_path / "raw"

    with pytest.raises(DatasetError, match="SHA-256"):
        acquire_dataset(output_dir, source=invalid_source)

    assert not output_dir.exists()


def test_malformed_source_leaves_no_dataset(tmp_path: Path) -> None:
    _, source = _source_archive(tmp_path, replacements={"ratings.dat": b"1::1::9::978300760\n"})
    output_dir = tmp_path / "raw"

    with pytest.raises(DatasetError, match="rating must be between 1 and 5"):
        acquire_dataset(output_dir, source=source)

    assert not output_dir.exists()


def test_unexpected_archive_member_is_rejected(tmp_path: Path) -> None:
    _, source = _source_archive(tmp_path, extra_member="../unexpected")

    with pytest.raises(DatasetError, match="expected MovieLens 1M files"):
        acquire_dataset(tmp_path / "raw", source=source)


def test_changed_existing_data_is_not_replaced(tmp_path: Path) -> None:
    _, source = _source_archive(tmp_path)
    output_dir = tmp_path / "raw"
    acquire_dataset(output_dir, source=source)
    changed = b"changed\n"
    (output_dir / "movies.dat").write_bytes(changed)

    with pytest.raises(DatasetError, match="not a valid reusable dataset"):
        acquire_dataset(output_dir, source=source)

    assert (output_dir / "movies.dat").read_bytes() == changed
