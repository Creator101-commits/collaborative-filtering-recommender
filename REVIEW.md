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
