"""Kaggle entrypoint for the MovieLens 1M frozen-protocol benchmark.

Installs ``recommender`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh``, downloads and verifies MovieLens 1M, prepares the
chronological splits, runs every validation configuration, freezes the
experiment card before any test read, runs the final comparison and report,
and writes ``benchmark/`` outputs including ``manifest.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"

MF_GRID = [
    {"latent_dim": 16, "learning_rate": 0.01, "regularization": 0.001},
    {"latent_dim": 32, "learning_rate": 0.01, "regularization": 0.001},
    {"latent_dim": 64, "learning_rate": 0.01, "regularization": 0.001},
    {"latent_dim": 32, "learning_rate": 0.005, "regularization": 0.01},
]
BPR_GRID = [
    {"latent_dim": 16, "learning_rate": 0.05, "regularization": 0.001},
    {"latent_dim": 32, "learning_rate": 0.05, "regularization": 0.001},
    {"latent_dim": 64, "learning_rate": 0.05, "regularization": 0.001},
    {"latent_dim": 32, "learning_rate": 0.01, "regularization": 0.001},
]
KNN_GRID = [
    {"n_neighbors": 40, "similarity": "cosine", "shrinkage": 10.0, "min_support": 1},
    {
        "n_neighbors": 40,
        "similarity": "adjusted_cosine",
        "shrinkage": 10.0,
        "min_support": 1,
    },
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "recommender").is_dir():
            return candidate.parent
    return None


def _install_package() -> tuple[Path, str]:
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError("no installable project under /kaggle/input")
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    info = proj / "run_info.json"
    sha = "unknown"
    if info.is_file():
        sha = str(json.loads(info.read_text()).get("git_sha", "unknown"))
    print(f"git_sha={sha}", flush=True)
    return proj, sha


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_configurations() -> dict[str, list[dict[str, Any]]]:
    return {
        "popularity": [{}],
        "user_knn": [dict(row) for row in KNN_GRID],
        "item_knn": [dict(row) for row in KNN_GRID],
        "matrix_factorization": [dict(row) for row in MF_GRID],
        "bpr": [dict(row) for row in BPR_GRID],
    }


def _run_validation(
    processed_dir: Path, tracking_dir: Path, artifact_dir: Path
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    list[dict[str, object]],
]:
    from recommender.experiments import (
        ExperimentConfig,
        run_bpr_baseline,
        run_matrix_factorization_baseline,
        run_neighborhood_baseline,
        run_popularity_baseline,
    )

    configurations = _validation_configurations()
    rows: list[dict[str, object]] = []
    per_model_rows: dict[str, list[dict[str, object]]] = {}
    for model_name, configs in configurations.items():
        per_model_rows[model_name] = []
        for config in configs:
            experiment = ExperimentConfig(seed=0)
            started = time.perf_counter()
            if model_name == "popularity":
                result = run_popularity_baseline(
                    processed_dir,
                    tracking_dir=tracking_dir,
                    artifact_dir=artifact_dir,
                    config=experiment,
                )
            elif model_name in {"user_knn", "item_knn"}:
                result = run_neighborhood_baseline(
                    processed_dir,
                    model=model_name,
                    n_neighbors=int(config["n_neighbors"]),
                    similarity=str(config["similarity"]),
                    shrinkage=float(config["shrinkage"]),
                    min_support=int(config["min_support"]),
                    tracking_dir=tracking_dir,
                    artifact_dir=artifact_dir,
                    config=experiment,
                )
            elif model_name == "matrix_factorization":
                result = run_matrix_factorization_baseline(
                    processed_dir,
                    latent_dim=int(config["latent_dim"]),
                    learning_rate=float(config["learning_rate"]),
                    regularization=float(config["regularization"]),
                    tracking_dir=tracking_dir,
                    artifact_dir=artifact_dir,
                    config=experiment,
                )
            else:
                result = run_bpr_baseline(
                    processed_dir,
                    latent_dim=int(config["latent_dim"]),
                    learning_rate=float(config["learning_rate"]),
                    regularization=float(config["regularization"]),
                    tracking_dir=tracking_dir,
                    artifact_dir=artifact_dir,
                    config=experiment,
                )
            row = {
                "model": model_name,
                "config": json.dumps(config, sort_keys=True),
                **{key: value for key, value in config.items()},
                "train_seconds": result.train_seconds,
                "inference_seconds": result.inference_seconds,
                "wall_seconds": time.perf_counter() - started,
                "validation_best_epoch": result.best_epoch,
                **result.metrics,
            }
            rows.append(row)
            per_model_rows[model_name].append(row)
            print(
                f"validation {model_name} {config}: "
                f"ndcg@10={result.metrics.get('ndcg@10', 0):.4f}",
                flush=True,
            )

    from recommender.experiments.protocol import (
        MODEL_HYPERPARAMETER_KEYS,
        build_model_configuration,
        select_configuration,
    )

    selected: dict[str, dict[str, object]] = {}
    selected_metrics: dict[str, dict[str, object]] = {}
    for model_name, candidates in per_model_rows.items():
        winner = select_configuration(candidates)
        best_epoch = winner.get("validation_best_epoch")
        selected[model_name] = build_model_configuration(
            model_name,
            winner,
            best_epoch=best_epoch if isinstance(best_epoch, int) else None,
        )
        config_keys = set(MODEL_HYPERPARAMETER_KEYS[model_name]) | {
            "model",
            "config",
            "train_seconds",
            "inference_seconds",
            "wall_seconds",
            "validation_best_epoch",
        }
        selected_metrics[model_name] = {
            key: value for key, value in winner.items() if key not in config_keys
        }
    return selected, selected_metrics, rows


def main() -> None:
    BENCH.mkdir(parents=True, exist_ok=True)
    proj, sha = _install_package()
    sys.path.insert(0, str(proj / "src"))

    from recommender.data import acquire_dataset, prepare_dataset
    from recommender.experiments import (
        generate_report,
        run_final_comparison,
        write_experiment_card,
    )
    from recommender.experiments.protocol import audit_comparison_inputs

    raw_dir = WORK / "raw" / "ml-1m"
    processed_dir = WORK / "processed" / "movielens-1m"
    acquire_dataset(raw_dir)
    manifest_path = prepare_dataset(raw_dir, processed_dir)
    data_manifest_sha = _sha256(manifest_path)
    print(f"processed manifest sha256={data_manifest_sha}", flush=True)

    tracking_dir = BENCH / "mlruns-validation"
    artifact_dir = BENCH / "artifacts-validation"
    selected, selected_metrics, validation_rows = _run_validation(
        processed_dir, tracking_dir, artifact_dir
    )

    validation_csv = BENCH / "validation-results.csv"
    if validation_rows:
        with validation_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(validation_rows[0]))
            writer.writeheader()
            writer.writerows(validation_rows)

    audit = audit_comparison_inputs(processed_dir, selected)
    card_path = BENCH / "experiment-card.json"
    write_experiment_card(
        card_path,
        audit.data_manifest_sha256,
        selected,
        git_revision=sha,
        selected_metrics=selected_metrics,
    )
    card_write_finished = time.time()
    card_sha = _sha256(card_path)
    print(f"experiment card sha256={card_sha}", flush=True)

    first_test_read = time.time()
    if not card_write_finished <= first_test_read:
        raise RuntimeError("experiment card was not written before the test read")
    final_tracking = BENCH / "mlruns-final"
    final_artifacts = BENCH / "artifacts-final"
    result = run_final_comparison(
        processed_dir,
        card_path,
        tracking_dir=final_tracking,
        artifact_dir=final_artifacts,
    )
    report = generate_report(
        result.results_path, result.paired_results_path, result.slices_path, BENCH / "report"
    )

    versions = {
        name: metadata.version(name)
        for name in ("recommender", "numpy", "pandas", "pyarrow", "torch", "mlflow", "scipy")
        if _installed(name)
    }
    try:
        import torch

        hardware = {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda": torch.cuda.is_available(),
        }
    except Exception:
        hardware = {"gpu": "unknown", "cuda": False}

    manifest = {
        "benchmark": "movielens-1m-frozen-protocol",
        "card_sha256": card_sha,
        "card_write_epoch": card_write_finished,
        "data_manifest_sha256": data_manifest_sha,
        "first_test_read_epoch": first_test_read,
        "git_sha": sha,
        "hardware": hardware,
        "leading_model": result.leading_model,
        "library_versions": versions,
        "outputs": {
            "paired_results": str(result.paired_results_path),
            "report": str(report.report_path),
            "results": str(result.results_path),
            "slices": str(result.slices_path),
            "validation": str(validation_csv),
        },
        "protocol_version": "phase-8-v2",
        "selected_configurations": selected,
    }
    (BENCH / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("benchmark complete", flush=True)


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


if __name__ == "__main__":
    main()
