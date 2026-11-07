> This document is a roadmap. See README.md for implemented behavior and verification limits; planned capabilities are not measured results.

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


### Optional Kaggle execution

The workflow can run on a Kaggle notebook instead of a local MacBook when more compute or a clean temporary environment is preferred. Kaggle must provide Python 3.12 because the project declares `>=3.12,<3.13`; check the notebook runtime before installing dependencies.

```python
!git clone https://github.com/Creator101-commits/collaborative-filtering-recommender.git
%cd collaborative-filtering-recommender
!python --version
!pip install -e .
!recommender download
!recommender prepare
```

Upload or create the validation-selected `experiment-card.json`, then run the frozen comparison and report commands:

```python
!recommender compare --experiment-card experiment-card.json
!recommender report
```

Kaggle Internet access must be enabled for the download command, or the validated raw dataset must be uploaded. The current implementation trains PyTorch models and runs SciPy KNN on the CPU; selecting a Kaggle GPU does not accelerate the benchmark unless device support is added. Save generated artifacts before the notebook session ends, and do not commit raw data, MLflow runs, checkpoints, or other large artifacts. This note documents an execution option only; it does not complete Phase 10 or replace the required clean-checkout verification.
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

## Phased delivery plan

Complete the phases in order. Do not start a phase until the previous phase's exit gate passes. Every phase includes its tests and documentation; a phase is not complete when its code exists but its checks fail.

**Current status:** Phases 1 through 9 are implemented and verified. Phase 10 is in progress: the report generator and release documentation are implemented; a clean-checkout workflow run, publication-ready narrative, and v1 release remain.

### Phase 1: Reproducible project skeleton
**Status:** Complete.

**Work**

- Create `pyproject.toml` and `uv.lock` for Python 3.12.
- Establish the `src/recommender/` package and `tests/` layout.
- Add a thin Typer CLI with a `recommender --help` entry point.
- Configure pytest, Ruff, mypy, and GitHub Actions.
- Add development and verification commands to the README.

**Exit gate**

- A clean checkout can run `uv sync`.
- `recommender --help`, tests, linting, and type checking pass locally and in CI.

### Phase 2: Dataset acquisition and source validation
**Status:** Complete.

**Depends on:** Phase 1.

**Work**

- Implement an idempotent MovieLens 1M download command using the official source.
- Verify the archive checksum and expected source files before extraction.
- Parse source records into explicit typed structures and reject malformed input.
- Record source URL, acquisition time, archive hash, file sizes, and record counts in a raw-data manifest.
- Test checksum failures, malformed records, repeat downloads, and tiny committed fixtures.

**Exit gate**

- One command produces a validated raw dataset without manual file movement.
- Repeating the command leaves valid existing data unchanged.
- Raw data and temporary download files are excluded from Git.

### Phase 3: Deterministic preparation and frozen splits
**Status:** Complete.

**Depends on:** Phase 2.

**Work**

- Resolve the positive-rating threshold and minimum-history rules before implementing the split.
- Create stable zero-based user and item indices.
- Build deterministic per-user chronological train, validation, and test splits.
- Hold out the last two positive events; keep every source interaction earlier than validation in train and exclude later unused events.
- Write typed, deterministically sorted Parquet tables and a processed-data manifest.
- Validate that partitions do not overlap, training events precede held-out events, and every evaluated user has eligible candidates.
- Test stable indices, split invariants, repeated-output hashes, and stale-manifest rejection.

**Exit gate**

- Repeated preparation from the same raw data and configuration produces identical artifacts.
- The manifest captures every filtering and split decision needed to reproduce the data.
- The test split is frozen and inaccessible to model selection.

### Phase 4: Evaluation foundation and popularity baseline
**Status:** Complete.

**Depends on:** Phase 3.

**Work**

- Define the shared recommender protocol and artifact metadata.
- Implement full-catalog candidate generation that removes only interactions unavailable under the declared protocol.
- Implement ranking, coverage, novelty, popularity, and concentration metrics.
- Verify each metric with small examples whose expected values are calculated by hand.
- Implement the popularity baseline and a common experiment runner with seeds, runtimes, Git revision, configuration, and data-manifest hashes.
- Log the first end-to-end validation run to local MLflow and generate a result table.

**Exit gate**

- One command runs acquisition through baseline evaluation without test-label access.
- Candidate and leakage invariants have direct tests.
- The baseline run is reproducible from its saved configuration and artifact metadata.

### Phase 5: Neighborhood models
**Status:** Complete.

**Depends on:** Phase 4.

**Work**

- Implement sparse user-KNN and item-KNN behind the shared protocol.
- Support only validation-justified settings for similarity, neighbor count, shrinkage, and minimum support.
- Test similarity, neighbor selection, seen-item filtering, scoring, and serialization on small fixtures.
- Run a bounded validation comparison against the popularity baseline.

**Exit gate**

- Both KNN models train, save, load, and evaluate through the common runner.
- Reloaded artifacts reproduce recommendations.
- Selected KNN settings and rejected alternatives are recorded without test-set access.

### Phase 6: Biased matrix factorization
**Status:** Complete.

**Depends on:** Phase 5.

**Work**

- Implement seeded PyTorch biased matrix factorization with mini-batch training.
- Add validation-based checkpointing and shared artifact serialization.
- Test deterministic initialization, gradient updates, save/load parity, and overfitting on a tiny synthetic fixture.
- Compare a bounded set of configurations on validation data.

**Exit gate**

- Training and evaluation are repeatable for a fixed seed.
- A reloaded checkpoint reproduces scores within a documented numerical tolerance.
- The selected configuration is recorded before test evaluation.

### Phase 7: Bayesian Personalized Ranking
**Status:** Complete.

**Depends on:** Phase 6.

**Work**

- Implement seeded negative sampling and BPR loss using the same interaction data and candidate catalog.
- Add validation-based checkpointing and shared artifact serialization.
- Test that negatives are unobserved under the training protocol, gradients update parameters, save/load preserves scores, and a tiny fixture can be overfit.
- Compare a bounded set of configurations on validation data.

**Exit gate**

- BPR trains and evaluates through the same runner and evaluator as every other model.
- Sampling is reproducible for a fixed seed and cannot draw held-out positives as training evidence.
- The selected configuration is recorded before test evaluation.

### Phase 8: Lock the comparison protocol
**Status:** Complete.

**Depends on:** Phase 7.

**Work**

- Resolve the final metric definitions, slice boundaries, seed list, bootstrap settings, and compute budget.
- Standardize MLflow fields and artifact layouts across all models.
- Audit configurations and manifests for candidate-set parity and leakage.
- Select one final configuration per model using validation results only.
- Write an experiment card that freezes the test procedure before opening test results.

**Exit gate**

- Every model can be reproduced from a configuration, data-manifest hash, code revision, and seed.
- The experiment card names all models, metrics, slices, seeds, and statistical comparisons.
- No unresolved protocol decision can change after test results are viewed.

### Phase 9: Final test evaluation and analysis
**Status:** Complete.

**Depends on:** Phase 8.

**Work**

- Evaluate every locked model once under the frozen test protocol, using the declared repeated seeds for learned models.
- Calculate aggregate metrics and 95% user-level bootstrap confidence intervals.
- Calculate paired per-user bootstrap differences between the leading model and each baseline.
- Produce history-length, item-popularity, cold-start-proxy, and recommendation-concentration slices.
- Record runtimes, practical memory measurements, artifact sizes, neutral results, failed runs, and representative failure cases.

**Exit gate**

- A consolidated machine-readable results table contains estimates, intervals, seeds, runtimes, and artifact references.
- Figures and tables directly answer every research question without unsupported winner claims.
- Final test results are never used to retune or reselect a model.

### Phase 10: Reproducible report and v1 release
**Status:** In progress.

**Depends on:** Phase 9.

**Work**

- Write the final report: abstract, motivation, methods, data, protocol, results, uncertainty, failure analysis, limitations, and conclusions.
- Generate publication-quality tables, figures, and machine-readable summarized results without retraining models.
- Replace README placeholders with verified installation, workflow, reproduction, attribution, and artifact instructions.
- Run the complete workflow from a clean checkout and document expected runtime and hardware.
- Decide where raw MLflow runs and large artifacts are published while keeping them out of Git.

**Exit gate**

- Tests, linting, typing, CI, the smoke experiment, and report generation all pass.
- Every advertised command has been verified from a clean environment.
- The report can be traced to immutable configurations, manifests, run IDs, and the Git revision.
- Tag the reproducible v1 release only after all earlier gates pass.

## Open questions

Resolve each decision by the named phase and record the outcome in configuration or project documentation before continuing:

1. **Positive relevance threshold (resolved in Phase 3):** The primary split uses ratings `>= 4`. Any `>= 3.5` or all-interaction variant is a preregistered sensitivity analysis, not a response to test results.
2. **Minimum user history (resolved in Phase 3):** Evaluated users need at least 3 positive interactions, leaving at least one training interaction plus validation and test events after chronological ordering. The processed manifest records the threshold, exclusion count, and split protocol.
3. **KNN similarity (resolved in Phase 5):** Implement cosine as the primary setting and adjusted cosine as the single validation alternative. Pearson is not supported; the evaluator and candidate protocol stay unchanged while the two settings are compared on validation runs.
4. **Learned-model objectives (resolved in Phases 6-7):** Use squared-error learning for biased matrix factorization and pairwise ranking for BPR. Do not add another objective unless validation evidence and the compute budget justify it.
5. **Bootstrap method (before Phase 8):** Use user-level resampling because the reported metrics are user aggregates, and document that it does not capture every dependency caused by shared items.
6. **Novelty definition (before Phase 8):** Choose one published inverse-popularity formulation and preserve it in configuration before final test evaluation.
7. **Demographic auditing (before Phase 8):** Leave it disabled by default. Enable it only for a precise research purpose with documented ethical interpretation and dataset limitations.
8. **Compute budget (resolved in Phase 5 for KNN):** Run at most one validation run for each of the two KNN axes and two similarity settings, using `n_neighbors=40`, `shrinkage=10`, `min_support=1`, and one seed. Do not inspect test labels during this comparison.
9. **Result publication (before Phase 10):** Decide whether summarized tables and figures are committed while raw MLflow runs remain external or are attached to a release.
