"""Command-line interface for the benchmark."""

from pathlib import Path

import typer

from recommender.data import (
    DEFAULT_MINIMUM_POSITIVE_HISTORY,
    DEFAULT_POSITIVE_RATING_THRESHOLD,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DatasetError,
    acquire_dataset,
    prepare_dataset,
)
from recommender.experiments import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_FINAL_ARTIFACT_DIR,
    DEFAULT_FINAL_TRACKING_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_TRACKING_DIR,
    ExperimentConfig,
    ExperimentError,
    ProtocolError,
    ReportError,
    generate_report,
    run_bpr_baseline,
    run_final_comparison,
    run_matrix_factorization_baseline,
    run_neighborhood_baseline,
    run_popularity_baseline,
)

app = typer.Typer(help="Run the reproducible MovieLens 1M benchmark.")


@app.callback()
def main() -> None:
    """Run the reproducible MovieLens 1M benchmark."""


@app.command()
def download(
    output_dir: Path = typer.Option(
        DEFAULT_RAW_DIR, help="Raw dataset destination.", file_okay=False
    ),
) -> None:
    """Download and validate the official MovieLens 1M dataset."""
    try:
        manifest = acquire_dataset(output_dir)
    except DatasetError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Dataset ready: {manifest}")


@app.command()
def prepare(
    raw_dir: Path = typer.Option(
        DEFAULT_RAW_DIR, help="Validated raw dataset directory.", file_okay=False
    ),
    output_dir: Path = typer.Option(
        DEFAULT_PROCESSED_DIR, help="Processed dataset destination.", file_okay=False
    ),
    positive_rating_threshold: int = typer.Option(
        DEFAULT_POSITIVE_RATING_THRESHOLD, help="Minimum rating treated as positive."
    ),
    minimum_positive_history: int = typer.Option(
        DEFAULT_MINIMUM_POSITIVE_HISTORY, help="Minimum positive interactions per evaluated user."
    ),
) -> None:
    """Prepare indexed Parquet tables and frozen chronological splits."""
    try:
        manifest = prepare_dataset(
            raw_dir,
            output_dir,
            positive_rating_threshold=positive_rating_threshold,
            minimum_positive_history=minimum_positive_history,
        )
    except (DatasetError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Processed dataset ready: {manifest}")


@app.command()
def evaluate(
    model: str = typer.Option(
        "popularity", help="Model: popularity, user_knn, item_knn, matrix_factorization, or bpr."
    ),
    processed_dir: Path = typer.Option(
        DEFAULT_PROCESSED_DIR, help="Prepared dataset directory.", file_okay=False
    ),
    tracking_dir: Path = typer.Option(
        DEFAULT_TRACKING_DIR, help="Local MLflow tracking directory.", file_okay=False
    ),
    artifact_dir: Path = typer.Option(
        DEFAULT_ARTIFACT_DIR, help="Model and result artifact directory.", file_okay=False
    ),
    cutoff: int = typer.Option(10, help="Recommendation cutoff."),
    positive_rating_threshold: int = typer.Option(
        DEFAULT_POSITIVE_RATING_THRESHOLD, help="Minimum rating counted as positive."
    ),
    n_neighbors: int = typer.Option(40, help="Number of nearest neighbors for KNN models."),
    similarity: str = typer.Option("cosine", help="KNN similarity setting."),
    shrinkage: float = typer.Option(10.0, help="KNN similarity shrinkage."),
    min_support: int = typer.Option(1, help="Minimum shared support for KNN neighbors."),
    aggregation: str = typer.Option(
        "weighted_mean", help="KNN score aggregation: weighted_mean or weighted_sum."
    ),
    latent_dim: int = typer.Option(32, help="Latent factor count for matrix factorization."),
    learning_rate: float = typer.Option(0.01, help="Matrix-factorization learning rate."),
    regularization: float = typer.Option(0.0001, help="Matrix-factorization L2 penalty."),
    epochs: int = typer.Option(20, help="Matrix-factorization training epochs."),
    batch_size: int = typer.Option(1024, help="Matrix-factorization mini-batch size."),
    negative_samples: int = typer.Option(1, help="Negative samples per positive BPR pair."),
    seed: int = typer.Option(0, help="Experiment seed."),
    experiment_name: str = typer.Option(
        DEFAULT_EXPERIMENT_NAME, help="Local MLflow experiment name."
    ),
) -> None:
    """Fit a selected baseline and evaluate against validation only."""
    try:
        config = ExperimentConfig(
            cutoff=cutoff,
            positive_rating_threshold=positive_rating_threshold,
            seed=seed,
            experiment_name=experiment_name,
        )
        if model == "popularity":
            result = run_popularity_baseline(
                processed_dir,
                tracking_dir=tracking_dir,
                artifact_dir=artifact_dir,
                config=config,
            )
        elif model == "bpr":
            result = run_bpr_baseline(
                processed_dir,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                regularization=regularization,
                epochs=epochs,
                batch_size=batch_size,
                negative_samples=negative_samples,
                tracking_dir=tracking_dir,
                artifact_dir=artifact_dir,
                config=config,
            )
        elif model == "matrix_factorization":
            result = run_matrix_factorization_baseline(
                processed_dir,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                regularization=regularization,
                epochs=epochs,
                batch_size=batch_size,
                tracking_dir=tracking_dir,
                artifact_dir=artifact_dir,
                config=config,
            )
        else:
            result = run_neighborhood_baseline(
                processed_dir,
                model=model,
                n_neighbors=n_neighbors,
                similarity=similarity,
                shrinkage=shrinkage,
                min_support=min_support,
                aggregation=aggregation,
                tracking_dir=tracking_dir,
                artifact_dir=artifact_dir,
                config=config,
            )
    except (ExperimentError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Validation run: {result.run_id}")
    typer.echo(f"Results: {result.results_path}")
    for name, value in result.metrics.items():
        typer.echo(f"{name}: {value:.6f}")


@app.command()
def compare(
    experiment_card: Path = typer.Option(
        Path("experiment-card.json"), help="Frozen experiment card.", file_okay=True, dir_okay=False
    ),
    processed_dir: Path = typer.Option(
        DEFAULT_PROCESSED_DIR, help="Prepared dataset directory.", file_okay=False
    ),
    tracking_dir: Path = typer.Option(
        DEFAULT_FINAL_TRACKING_DIR, help="Local MLflow tracking directory.", file_okay=False
    ),
    artifact_dir: Path = typer.Option(
        DEFAULT_FINAL_ARTIFACT_DIR, help="Final result artifact directory.", file_okay=False
    ),
) -> None:
    """Evaluate the frozen model comparison once against the test split."""
    try:
        result = run_final_comparison(
            processed_dir,
            experiment_card,
            tracking_dir=tracking_dir,
            artifact_dir=artifact_dir,
        )
    except (ExperimentError, ProtocolError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Leading model: {result.leading_model}")
    typer.echo(f"Results: {result.results_path}")
    typer.echo(f"Paired results: {result.paired_results_path}")
    typer.echo(f"Slices: {result.slices_path}")


@app.command()
def report(
    results_path: Path = typer.Option(
        DEFAULT_FINAL_ARTIFACT_DIR / "final-results.csv",
        help="Final aggregate result table.",
        file_okay=True,
        dir_okay=False,
    ),
    paired_results_path: Path = typer.Option(
        DEFAULT_FINAL_ARTIFACT_DIR / "paired-results.csv",
        help="Paired comparison table.",
        file_okay=True,
        dir_okay=False,
    ),
    slices_path: Path = typer.Option(
        DEFAULT_FINAL_ARTIFACT_DIR / "final-slices.csv",
        help="Final slice table.",
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_REPORT_DIR, help="Report output directory.", file_okay=False
    ),
) -> None:
    """Generate the report from final CSV artifacts without retraining."""
    try:
        result = generate_report(
            results_path, paired_results_path, slices_path, output_dir
        )
    except ReportError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Report: {result.report_path}")
    typer.echo(f"Summary: {result.summary_path}")
    typer.echo(f"Figure: {result.figure_path}")
