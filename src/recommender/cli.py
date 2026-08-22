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
