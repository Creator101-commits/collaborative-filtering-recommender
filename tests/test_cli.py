"""Tests for the command-line interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from recommender.cli import app
from recommender.experiments import ExperimentResult


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


def test_evaluate_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = ExperimentResult(
        run_id="run-id",
        metrics={"recall@10": 0.5},
        model_path=tmp_path / "model.json",
        results_path=tmp_path / "results.csv",
        tracking_uri="file:///tmp/mlruns",
        data_manifest_sha256="hash",
        git_revision="revision",
        train_seconds=0.1,
        inference_seconds=0.1,
    )

    def fake_run(processed_dir: Path, **kwargs: object) -> ExperimentResult:
        assert processed_dir == tmp_path / "processed"
        return result

    monkeypatch.setattr("recommender.cli.run_popularity_baseline", fake_run)
    cli_result = CliRunner().invoke(
        app,
        ["evaluate", "--processed-dir", str(tmp_path / "processed")],
    )

    assert cli_result.exit_code == 0
    assert "Validation run: run-id" in cli_result.output
    assert "recall@10: 0.500000" in cli_result.output


def test_evaluate_knn_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = ExperimentResult(
        run_id="knn-run",
        metrics={},
        model_path=tmp_path / "model.json",
        results_path=tmp_path / "results.csv",
        tracking_uri="file:///tmp/mlruns",
        data_manifest_sha256="hash",
        git_revision="revision",
        train_seconds=0.1,
        inference_seconds=0.1,
    )
    captured: dict[str, object] = {}

    def fake_run(processed_dir: Path, **kwargs: object) -> ExperimentResult:
        captured["processed_dir"] = processed_dir
        captured.update(kwargs)
        return result

    monkeypatch.setattr("recommender.cli.run_neighborhood_baseline", fake_run)
    cli_result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--model",
            "user_knn",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--n-neighbors",
            "3",
            "--shrinkage",
            "2.0",
            "--min-support",
            "2",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["processed_dir"] == tmp_path / "processed"
    assert captured["model"] == "user_knn"
    assert captured["n_neighbors"] == 3
    assert captured["shrinkage"] == 2.0
    assert captured["min_support"] == 2


def test_evaluate_matrix_factorization_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = ExperimentResult(
        run_id="mf-run",
        metrics={},
        model_path=tmp_path / "model.pt",
        results_path=tmp_path / "results.csv",
        tracking_uri="file:///tmp/mlruns",
        data_manifest_sha256="hash",
        git_revision="revision",
        train_seconds=0.1,
        inference_seconds=0.1,
    )
    captured: dict[str, object] = {}

    def fake_run(processed_dir: Path, **kwargs: object) -> ExperimentResult:
        captured["processed_dir"] = processed_dir
        captured.update(kwargs)
        return result

    monkeypatch.setattr("recommender.cli.run_matrix_factorization_baseline", fake_run)
    cli_result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--model",
            "matrix_factorization",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--latent-dim",
            "8",
            "--epochs",
            "4",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["processed_dir"] == tmp_path / "processed"
    assert captured["latent_dim"] == 8
    assert captured["epochs"] == 4

def test_evaluate_bpr_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = ExperimentResult(
        run_id="bpr-run",
        metrics={},
        model_path=tmp_path / "model.pt",
        results_path=tmp_path / "results.csv",
        tracking_uri="file:///tmp/mlruns",
        data_manifest_sha256="hash",
        git_revision="revision",
        train_seconds=0.1,
        inference_seconds=0.1,
    )
    captured: dict[str, object] = {}

    def fake_run(processed_dir: Path, **kwargs: object) -> ExperimentResult:
        captured["processed_dir"] = processed_dir
        captured.update(kwargs)
        return result

    monkeypatch.setattr("recommender.cli.run_bpr_baseline", fake_run)
    cli_result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--model",
            "bpr",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--latent-dim",
            "8",
            "--epochs",
            "4",
            "--negative-samples",
            "2",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["processed_dir"] == tmp_path / "processed"
    assert captured["latent_dim"] == 8
    assert captured["epochs"] == 4
    assert captured["negative_samples"] == 2
