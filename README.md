# Collaborative Filtering Recommender

Which recommender actually wins on real ratings? This project benchmarks
popularity, neighborhood and latent-factor models on MovieLens 1M:
candidates are selected on validation, the test set is read once, and
every metric carries a bootstrap interval.

![Collaborative filtering benchmark results: test recall and ndcg, recall versus K, recall versus catalog coverage, KNN validation aggregation and popularity bias](image.png)

The figure follows the comparison: (a) test recall@10 and ndcg@10 per
model, (b) recall versus K, (c) recall versus catalog coverage, (d) the
winning KNN aggregation, (e) how popular the picks are.

## Results

| model | seeds | recall@10 [CI] | ndcg@10 | mean_popularity@10 | catalog_coverage@10 |
| --- | ---: | --- | --- | ---: | ---: |
| popularity | 1 | 0.0442 [0.0391, 0.0495] | 0.0222 | 2065.7 | 0.046 |
| user_knn | 1 | 0.0978 [0.0903, 0.1054] | 0.0491 | 1373.6 | 0.210 |
| item_knn | 1 | 0.0981 [0.0906, 0.1057] | 0.0491 | 1073.5 | 0.329 |
| matrix_factorization | 3 | 0.0414-0.0419 | 0.0191-0.0194 | ~1992 | 0.046 |
| bpr | 3 | 0.0630-0.0713 | 0.0313-0.0345 | ~1216 | ~0.336 |

- **The two KNN variants tie for first.** Paired recall@10 difference
  -0.0003 [-0.0081, +0.0075]; both roughly double popularity's 0.0442.
- **BPR is the strongest learned model.** recall@10 0.0630-0.0713 across
  seeds 0-2 (mean 0.0676), above popularity, below KNN.
- **Matrix factorization collapses to popularity level.** recall@10
  0.0414-0.0419, mean_popularity@10 ~1992 vs 2066, coverage 0.046: it
  learned the popular head.
- **The aggregation choice was decisive on validation.** weighted_sum
  recall@10 0.0820-0.1112 vs 0.0003-0.0028 for weighted_mean (29x-371x
  per cell); this validation-only evidence motivated phase-8-v3.
- **KNN and BPR diversify; popularity and MF do not.** coverage@10
  0.21-0.33 vs 0.046, and mean_popularity@10 is lower for KNN
  (1074-1374) and BPR (~1216) than popularity's 2066.
- **Popularity is a strong baseline, not the ceiling.** recall@10
  0.0442; every other family except MF beats it.

## How it works

```text
download + pinned checksum verify -> ratings >= 4 positive, >= 3 per user
-> per-user chronological split; last interactions -> validation/test
-> evaluate fits on train.parquet only, ranks catalog, selects on validation
-> freeze experiment card -> compare does one test read -> bootstrap CIs
```

Validation selection chose `cosine` similarity with `weighted_sum`
aggregation for both KNN families (`n_neighbors=40`, `shrinkage=10`,
`min_support=1`).

## Run it

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run mypy src

# full pipeline (Kaggle reproduction: scripts/kaggle_run.sh)
uv run recommender download
uv run recommender prepare
uv run recommender evaluate --model user_knn   # repeat per model
# after writing experiment-card.json from validation results:
uv run recommender compare --experiment-card experiment-card.json
uv run recommender report
```

`evaluate` takes `--model` (popularity, user_knn, item_knn,
matrix_factorization, bpr); `EXPERIMENT_CARD.md` froze the protocol
before the test read.

## Notes

- The reported leader is descriptive from test scores, so paired
  intervals against it are exploratory and unadjusted for selection.
- KNN and popularity use one seed; MF and BPR use three. Per-user
  chronology means there is no single global time cutoff.
- Protocol history is disclosed in `EXPERIMENT_CARD.md`: phase-8-v2 ran
  to completion before the KNN aggregation defect was found, and its
  test read was not used for selection.
- Data: MovieLens 1M (GroupLens, research use with attribution);
  ratings >= 4 count as positive and evaluated users need >= 3 positive
  interactions.
- Protocol `phase-8-v3`; curated kernel output is in `results/` (git
  SHA in `manifest.json`), and `image.png` is rendered from it.
