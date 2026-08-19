# Build Plan: Collaborative Filtering Recommender

## Goal

Build a reproducible Python research project that determines how several collaborative-filtering approaches perform under the same top-N recommendation protocol. The finished project will explain not only which model scores highest, but also how accuracy, catalog coverage, popularity bias, cold-start behavior, training cost, and statistical uncertainty change across methods.

## Scope

### Included in v1

- Acquire and validate the MovieLens 1M dataset through a repeatable command.
- Convert the source files into typed Parquet tables with a documented schema.
- Construct a deterministic per-user chronological train/validation/test split.
- Prevent future interactions from leaking into training features or candidate generation.
- Implement one non-personalized popularity baseline.
- Implement user-based and item-based nearest-neighbor collaborative filtering.
- Implement a biased matrix-factorization model in PyTorch.
- Implement a pairwise-ranking model using Bayesian Personalized Ranking loss in PyTorch.
- Generate top-N recommendations over the complete eligible movie catalog.
- Evaluate all models using the same candidate set and metrics.
- Report Recall@K, NDCG@K, MRR@K, hit rate, catalog coverage, novelty, and popularity concentration.
- Analyze results by user-history length, item popularity, and model family.
- Quantify uncertainty with user-level bootstrap confidence intervals and repeated random seeds for learned models.
- Track configurations, metrics, runtimes, environment information, and artifacts with local MLflow.
- Produce publication-quality tables, plots, an experiment card, and a final research report.
- Provide a command-line workflow that recreates the main results from a clean checkout.
- Add automated unit and integration tests plus linting and static type checking.

### Explicitly outside v1

- Production deployment, online serving, authentication, and cloud infrastructure.
- A Streamlit or web interface.
- Live user feedback or online A/B testing.
- Deep sequential, graph-neural-network, transformer, or large-language-model recommenders.
- Content-based recommendations derived from posters, plots, reviews, or external metadata.
- Reinforcement learning and bandit algorithms.
- Multiple production-scale datasets.
- Claims about real-world business lift based only on offline MovieLens results.
- Automated hyperparameter searches large enough to require paid compute.
- Training on protected demographic fields. Demographic columns may be used only for optional auditing if their ethical use and limitations are documented.

## Research questions

1. How much do personalized collaborative-filtering models improve top-N ranking quality over a popularity baseline?
2. Do neighborhood methods and latent-factor methods fail on the same users and items?
3. What accuracy trade-offs appear when models recommend less-popular items and cover more of the catalog?
4. How does performance change for users with short, medium, and long interaction histories?
5. Are observed metric differences larger than their bootstrap uncertainty intervals?

## Stack

- **Language:** Python 3.12
- **Environment and packaging:** `uv` with a `pyproject.toml` and locked dependencies
- **Core numerical work:** NumPy, pandas, SciPy, and scikit-learn
- **Learned models:** PyTorch
- **Configuration:** YAML loaded into validated Python configuration objects
- **Data storage:** source text files in `data/raw/`; typed Parquet tables in `data/processed/`
- **Experiment tracking:** MLflow with a local file or SQLite-backed store
- **Visualization:** Matplotlib and seaborn
- **Command line:** Typer
- **Testing and quality:** pytest, pytest-cov, Ruff, and mypy
- **Automation:** GitHub Actions for tests, linting, type checking, and a small smoke experiment
- **Documentation:** Markdown research report plus generated CSV/LaTeX-ready result tables

## Architecture

```text
Dataset source
    |
    v
download + checksum verification
    |
    v
schema validation + Parquet conversion
    |
    v
chronological split + sparse interaction matrices
    |
    +----------------------+----------------------+------------------+
    |                      |                      |                  |
popularity            user/item KNN        matrix factorization    BPR
    |                      |                      |                  |
    +----------------------+----------------------+------------------+
                                   |
                                   v
                       shared top-N recommendation API
                                   |
                                   v
                  evaluator + bootstrap + slice analysis
                                   |
                                   v
                      MLflow logs + tables + figures
                                   |
                                   v
                           final research report
```

### Components

#### `src/recommender/data/`

Downloads the dataset, verifies its checksum, parses source files, validates schemas, creates stable integer indices, performs the chronological split, and writes processed artifacts. Acquisition must be idempotent: rerunning it must not silently replace a different file or create different processed records.

#### `src/recommender/models/`

Defines a common recommender protocol with `fit`, `score`, `recommend`, `save`, and `load` behavior. Individual modules implement popularity, user-KNN, item-KNN, biased matrix factorization, and BPR while exposing scores in a form understood by one evaluator.

#### `src/recommender/evaluation/`

Creates each user's eligible candidate catalog, filters training interactions, computes ranking and beyond-accuracy metrics, performs user-level bootstrap resampling, and calculates slice results. Metric functions must have direct tests using small examples whose correct answers can be calculated by hand.

#### `src/recommender/experiments/`

Loads configurations, sets seeds, invokes training and evaluation, captures runtimes and environment metadata, and logs outputs to MLflow. It must reject incomplete configurations and record the Git revision when available.

#### `src/recommender/reporting/`

Reads completed runs and produces model-comparison tables, confidence-interval plots, popularity-versus-accuracy plots, history-length slices, runtime charts, and machine-readable result files. Reporting must not retrain or mutate models.

#### `src/recommender/cli.py`

Provides stable `download`, `prepare`, `train`, `evaluate`, `compare`, and `report` commands. CLI handlers should remain thin and call tested modules rather than contain data or modeling logic.

### Communication between components

- Data modules write versioned Parquet tables and a JSON manifest containing hashes, record counts, schema information, split rules, and index mappings.
- Model modules consume processed tables or sparse matrices and produce a versioned model artifact plus run metadata.
- The evaluator consumes a model through the shared recommender protocol and emits long-form metric records keyed by model, run, seed, split, cutoff, and analysis slice.
- MLflow stores parameters, summary metrics, environment metadata, plots, tables, and paths to local model artifacts.
- Reporting consumes evaluator output and MLflow run metadata; it never accesses hidden test labels during model selection.

## Data

### Primary dataset

Use **MovieLens 1M** as the only required v1 dataset. It is small enough for local iteration and complete-catalog ranking while still supporting user-based, item-based, and latent-factor experiments.

### Source tables

The acquisition step is expected to parse these source entities:

- `ratings`: `user_id`, `movie_id`, `rating`, `timestamp`
- `movies`: `movie_id`, `title`, `genres`
- `users`: `user_id`, `gender`, `age_group`, `occupation`, `zip_code`

Only identifiers, ratings, timestamps, titles, and genres are necessary for v1. User demographic attributes will not be model inputs. If demographic auditing is later enabled, the report must explain the dataset's age, limitations, and why each slice is being inspected.

### Acquisition

- Implement a `download` command that fetches the official archive from GroupLens.
- Record the source URL, acquisition date, archive size, and a cryptographic checksum in the data manifest.
- Refuse to continue when a downloaded archive fails checksum or source-file validation.
- Preserve source files unchanged under `data/raw/movielens-1m/`.
- Keep downloaded files out of Git.
- Document the dataset license and required attribution before distributing any derived artifact.

### Processed schema

- `ratings.parquet`: typed original interactions plus stable zero-based user and item indices
- `movies.parquet`: movie identifiers, titles, parsed genre lists, and item indices
- `users.parquet`: user identifiers and user indices; demographics retained only for documented audit use
- `train.parquet`, `validation.parquet`, and `test.parquet`: interaction partitions with an explicit split label
- `manifest.json`: source hashes, transformation version, schemas, record counts, filtering decisions, time ranges, and split parameters

### Split protocol

- Sort interactions chronologically within each user.
- Keep users only when enough interactions remain to contribute to training, validation, and testing; make the minimum configurable and report how many users were removed.
- Reserve the last eligible positive interaction for test and the previous eligible positive interaction for validation.
- Treat ratings at or above a configurable threshold, initially `4`, as positive relevance for top-N evaluation.
- Use only interactions earlier than a held-out event as training evidence for that user.
- Exclude items already observed in training from the recommendation candidate set.
- Rank against the complete eligible item catalog by default rather than using convenient sampled negatives.
- Freeze the test split after preparation. Select models and hyperparameters using training and validation data only.

### Storage and reproducibility

Processed data will use Parquet with explicit dtypes and deterministic sorting. Every processed directory will contain a manifest sufficient to detect stale or incompatible outputs. Large data, checkpoints, and MLflow artifacts remain local and ignored by Git; configurations, source code, small test fixtures, and final summarized results remain version controlled.

## Evaluation design

### Primary metrics

- Recall@10
- NDCG@10
- MRR@10

### Secondary metrics

- Hit Rate@10
- Recall and NDCG at cutoffs 5 and 20
- Catalog coverage@10
- Mean recommended-item popularity
- Novelty based on inverse item popularity
- Gini concentration of recommendation exposure
- Training time, inference time, peak memory when practical, and artifact size

### Statistical reporting

- Calculate 95% user-level bootstrap confidence intervals for aggregate ranking metrics.
- Run learned models with at least three documented random seeds for the final comparison.
- Report paired per-user bootstrap differences between the leading model and each baseline.
- Treat small overlapping differences as uncertain rather than declaring a winner from rounded means.
- Preserve unsuccessful and neutral experiments in MLflow so the final report does not contain only favorable runs.

### Slice analysis

- User history length: short, medium, and long histories based on documented quantiles
- Item popularity: head, middle, and tail groups based on training interactions only
- Recommendation concentration: proportion of exposure assigned to the most popular 1%, 5%, and 10% of items
- Cold-start proxy: users with the shortest permitted histories and items with the fewest training interactions

## Milestones

### 1. Reproducible Python skeleton

Create `pyproject.toml`, lock dependencies, establish the `src/` package layout, add the Typer CLI, configure Ruff and mypy, and set up pytest and GitHub Actions. Ship a `recommender --help` command, one passing test, and CI that verifies a clean installation.

### 2. Dataset acquisition and validation

Implement the idempotent MovieLens downloader, checksum handling, source parser, schema validation, and raw-data manifest. Test parsers with tiny committed fixtures and test failure behavior for malformed records and invalid hashes. Ship a command that produces a validated raw dataset without manual file movement.

### 3. Deterministic preparation pipeline

Implement identifier indexing, rating-threshold configuration, per-user chronological splitting, Parquet output, and the processed manifest. Add invariant tests proving that partitions do not overlap, indices remain stable, training events precede held-out events, and repeated preparation produces identical hashes.

### 4. Shared evaluator and popularity baseline

Implement the recommender protocol, full-catalog candidate generation, ranking metrics, coverage and popularity metrics, and the popularity model. Verify metrics on hand-computed fixtures. Ship the first complete run from raw data through a tracked baseline and generated result table.

### 5. Neighborhood collaborative filtering

Implement sparse user-KNN and item-KNN with configurable similarity, neighbor count, shrinkage, and minimum support. Add unit tests for similarity and scoring, validation experiments for hyperparameter selection, and slice reports showing behavior across user-history and item-popularity groups.

### 6. Latent-factor models

Implement biased matrix factorization and BPR in PyTorch with seeded initialization, mini-batch training, validation-based checkpointing, and model serialization. Add gradient and overfitting tests on small synthetic data. Ship repeatable training and evaluation commands for both models.

### 7. Experiment tracking and controlled comparison

Standardize MLflow logging across all models, record data-manifest hashes and Git revisions, run bounded validation sweeps, select final configurations without test access, then evaluate selected models on the frozen test split across repeated seeds. Ship a consolidated results table containing means, intervals, runtimes, and artifact references.

### 8. Robustness, uncertainty, and failure analysis

Add bootstrap confidence intervals, paired model-difference estimates, history-length slices, popularity slices, recommendation-concentration analysis, and representative failure cases. Ship figures and tables that directly answer each research question, including results that do not support the expected conclusion.

### 9. Final research artifact

Write the final report with abstract, motivation, related-method overview, data, protocol, models, results, limitations, reproducibility instructions, and conclusions. Update the README so every advertised command has been verified from a clean environment. Tag a reproducible v1 release only after tests, linting, typing, the smoke experiment, and report generation pass.

## Open questions

These decisions do not block the skeleton or data milestones, but they must be resolved before final experiments:

1. **Positive relevance threshold:** Start with ratings `>= 4`; test `>= 3.5` or all non-negative interactions only as a documented sensitivity analysis, not after seeing test results.
2. **Minimum user history:** Select a minimum that guarantees train, validation, and test evidence while retaining most users. Record the exact rule and exclusion count in the manifest.
3. **KNN similarity:** Compare cosine and adjusted-cosine or Pearson-style similarity on validation data while keeping the full-catalog evaluator unchanged.
4. **Matrix-factorization objective:** Use squared-error learning for the biased factorization baseline and pairwise ranking for BPR. Decide whether an additional implicit-feedback pointwise objective adds enough information to justify another model.
5. **Bootstrap method:** Begin with resampling users because the reported metrics are user aggregates. Document that this does not capture every dependency caused by shared items.
6. **Novelty definition:** Choose one published, clearly stated inverse-popularity formulation before running final test evaluation and preserve it in configuration.
7. **Demographic auditing:** Leave disabled by default. Enable only if there is a precise research purpose, appropriate interpretation, and explicit discussion of sensitive attributes and dataset limitations.
8. **Compute budget:** Establish maximum epochs, validation trials, and wall-clock limits before hyperparameter tuning so comparisons remain feasible and reproducible on a normal laptop.
9. **Result publication:** Decide whether summarized result tables and small figures will be committed while raw MLflow runs remain external or attached to a release.
