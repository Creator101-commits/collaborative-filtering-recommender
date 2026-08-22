"""Tests for the command-line interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from recommender.cli import app


def test_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Run the reproducible MovieLens 1M benchmark." in result.output


def test_download_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    manifest = output_dir / "manifest.json"

    def fake_acquire(requested_output: Path) -> Path:
        assert requested_output == output_dir
        return manifest

    monkeypatch.setattr("recommender.cli.acquire_dataset", fake_acquire)
    result = CliRunner().invoke(app, ["download", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert str(manifest) in result.output


def test_prepare_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    manifest = output_dir / "manifest.json"

    def fake_prepare(
        requested_raw: Path,
        requested_output: Path,
        *,
        positive_rating_threshold: int,
        minimum_positive_history: int,
    ) -> Path:
        assert requested_raw == raw_dir
        assert requested_output == output_dir
        assert positive_rating_threshold == 4
        assert minimum_positive_history == 3
        return manifest

    monkeypatch.setattr("recommender.cli.prepare_dataset", fake_prepare)
    result = CliRunner().invoke(
        app,
        ["prepare", "--raw-dir", str(raw_dir), "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert str(manifest) in result.output
