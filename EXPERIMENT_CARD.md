# MovieLens 1M experiment card

Status: frozen before final test evaluation.

This card is the Phase 8 comparison contract. The test split must not be read while selecting a model, configuration, seed, metric, slice, or analysis method.

## Data and candidate protocol

- Dataset: MovieLens 1M.
- Relevance: a held-out rating is relevant when its rating is at least 4.
- Training: all source interactions earlier than validation for each evaluated user.
- Candidates: every indexed movie except that user's training items. Validation and test items remain eligible candidates.
- Demographics are not model features or default analysis slices.
- The processed-data manifest SHA-256 is recorded in the generated JSON card and every artifact.

## Models

The comparison contains exactly these model families:

- popularity
- user-KNN
- item-KNN
- biased matrix factorization
- Bayesian Personalized Ranking

One configuration per model is selected from validation results only. The selected configurations, data-manifest hash, protocol version, Git revision, and seed are written by `write_experiment_card` before test evaluation. A card with different contents cannot overwrite an existing card.

## Metrics

The primary cutoff is 10. Results are also reported at cutoffs 5 and 20.

- Recall, NDCG, MRR, and hit rate use binary held-out relevance and macro-average over users.
- Catalog coverage is the fraction of all catalog items exposed at least once in top-K lists.
- Mean recommended popularity is the mean number of positive training interactions per top-K exposure.
- Novelty is mean smoothed self-information in bits:
  `-log2((positive_training_count(item) + 1) / (total_positive_training_count + catalog_size))`.
- Gini exposure measures concentration of top-K exposure across the complete item catalog.

The novelty measure follows the self-information formulation used by Zhou et al., “Solving the apparent diversity-accuracy dilemma of recommender systems,” PNAS 2010: <https://doi.org/10.1073/pnas.1000488107>. Laplace smoothing is fixed here so zero-count catalog items remain finite and eligible.

## Uncertainty and slices

- Learned models use seeds 0, 1, and 2 in the final comparison.
- Aggregate ranking estimates use 10,000 user-level bootstrap replicates and percentile 95% intervals.
- Paired bootstrap differences compare the leading model with every baseline using the same resampled users.
- History slices are short, middle, and long based on the 1/3 and 2/3 quantiles of training interaction counts.
- Item slices are tail, middle, and head based on the 20th and 80th percentiles of positive training interaction counts.
- The cold-start proxy is the shortest permitted user history and the lowest-training-count items.
- Recommendation concentration reports exposure assigned to the most popular 1%, 5%, and 10% of items.

## Compute budget

- Popularity has one validation configuration.
- Each KNN family compares cosine and adjusted cosine only, with `n_neighbors=40`, `shrinkage=10`, and `min_support=1`.
- Each learned model may compare at most four validation configurations using one documented seed; the final selected configuration is then evaluated with seeds 0, 1, and 2.
- No hyperparameter search or re-selection is permitted after test results are viewed.

## Reproduction fields

Every run and model artifact records `protocol_version=phase-8-v1`, model name, configuration, seed, data-manifest SHA-256, Git revision, split, cutoff, training time, inference time, metrics, and artifact paths. The pre-test audit verifies model-set parity, the frozen split policy, and a single shared validation candidate catalog derived from training rows only.
