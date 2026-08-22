# Collaborative Filtering Recommender

> A reproducible MovieLens 1M benchmark for collaborative-filtering recommendation methods.

[![CI](https://github.com/Creator101-commits/collaborative-filtering-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Creator101-commits/collaborative-filtering-recommender/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/Creator101-commits/collaborative-filtering-recommender)](./LICENSE)

This project will compare popularity, user- and item-based KNN, biased matrix factorization, and Bayesian Personalized Ranking under one leakage-controlled top-N evaluation protocol. See [`PLAN.md`](./PLAN.md) for the phased research and implementation plan.

## Table of Contents

- [Collaborative Filtering Recommender](#collaborative-filtering-recommender)
- [Quickstart / Demo](#quickstart--demo)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [Release History](#release-history)
- [License](#license)
- [Meta](#meta)

## Quickstart / Demo
[(Back to top)](#table-of-contents)
Download, validate, and prepare MovieLens 1M with the reproducible workflow:

```sh
uv sync
uv run recommender download
uv run recommender prepare
```

## Installation
[(Back to top)](#table-of-contents)

Requirements:

- macOS or Linux
- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
git clone https://github.com/Creator101-commits/collaborative-filtering-recommender.git
cd collaborative-filtering-recommender
uv sync --locked
```

## Usage
[(Back to top)](#table-of-contents)

```sh
uv run recommender download
uv run recommender prepare
```

The `download` command fetches from GroupLens, verifies the pinned archive size and checksums, validates every source record, and writes `data/raw/movielens-1m/manifest.json`. Rerunning it reuses the data only when every file still matches the manifest. Use `--output-dir` to choose another location.

The `prepare` command reads the validated raw manifest, assigns stable zero-based indices in ascending identifier order, treats ratings `>= 4` as positive, and requires at least 3 positive interactions per evaluated user. It writes typed `ratings.parquet`, `movies.parquet`, `users.parquet`, `train.parquet`, `validation.parquet`, and `test.parquet` files plus `data/processed/movielens-1m/manifest.json`. Positive interactions are ordered by timestamp, movie ID, and source row; the last two become validation and test. `train.parquet` contains every source interaction earlier than validation, including lower-rated observations, while later unselected interactions remain outside training. Re-running with the same raw manifest and configuration reuses identical output and rejects stale or modified artifacts.

Use `--raw-dir`, `--output-dir`, `--positive-rating-threshold`, or `--minimum-positive-history` to override the defaults.
MovieLens 1M is provided by GroupLens for research use under conditions that require attribution and restrict redistribution and commercial use. Review the [dataset README and license](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt) before use. Publications should cite F. Maxwell Harper and Joseph A. Konstan, “The MovieLens Datasets: History and Context,” ACM TiiS 5(4), 2015, https://doi.org/10.1145/2827872.

Training, evaluation, comparison, and reporting commands are planned but not implemented yet.

## Development
[(Back to top)](#table-of-contents)

Run the same checks used by CI:

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run mypy src
```

## Contributing
[(Back to top)](#table-of-contents)

1. Fork the repository.
2. Create a focused feature branch.
3. Add tests for the change.
4. Run all development checks.
5. Open a pull request with a clear description of why the change is needed.

Use conventional commit messages such as `feat:`, `fix:`, and `chore:`.

## Release History
[(Back to top)](#table-of-contents)

No releases have been published yet.

## License
[(Back to top)](#table-of-contents)

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Meta
[(Back to top)](#table-of-contents)

Sreeharsha Kannegundla

Project: [github.com/Creator101-commits/collaborative-filtering-recommender](https://github.com/Creator101-commits/collaborative-filtering-recommender)
