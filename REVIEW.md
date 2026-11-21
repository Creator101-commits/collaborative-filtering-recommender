# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Fixed item-KNN seen-item filtering, safer tensor artifact loading, duplicate ranking rejection, numeric report validation, and Matplotlib report generation.

## Verified locally

- 66 tests passed with the repository's existing Python environment.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed.
- The README image was rendered from code using Matplotlib and visually inspected. Its inputs/results and a source fingerprint are in `scripts/figure-data.json`.

## Limits

The README figure is a synthetic four-user demonstration. Full MovieLens runs require acquired data and a frozen card; test-leading paired intervals are exploratory.

Kaggle API access failed because the execution environment could not resolve its hostname. The private notebook was also unavailable through the browser. No GPU training or full-scale external benchmark result is claimed by this review. New CI workflows are configuration changes; their hosted execution must be checked after publication.

## Protocol v2 correction

`phase-8-v2` replaces `phase-8-v1` before any test data was read for selection or evaluation. The final test comparison previously refit on `train.parquet` (interactions earlier than validation), leaving each user's validation positive out of training while it remained an eligible candidate. `prepare` now emits `train_for_test.parquet` with every interaction strictly earlier than the test interaction, `run_final_comparison` fits on it, and `audit_comparison_inputs` asserts the containment and ordering invariants. Unit tests cover each assertion on the small fixture.

## Protocol v3 correction

`phase-8-v3` replaces `phase-8-v2` after the v2 Kaggle run (git df2c8f7) completed, including the frozen test read. Reviewing that run showed user-KNN and item-KNN validation recall@10 between 0.0003 and 0.003 with mean_popularity@10 around 150 to 250, versus 0.053 recall@10 for popularity, 0.078 for BPR, and mean_popularity@10 near 2000 for the other models. The cause was the score aggregation: the weighted mean used for top-N ranking is a rating-prediction weighted mean that lets an item reached through one similar neighbor outrank items supported by many neighbors. `NeighborhoodConfig` gains an `aggregation` setting (`weighted_mean` unchanged, `weighted_sum` returning the unnormalized similarity-times-rating sum), threaded through the CLI, runners, final comparison, and artifacts; the protocol allows the key and the Kaggle grid crosses both similarities with both aggregations. The correction rests on validation-only evidence; the completed v2 test read was not used for selection.
