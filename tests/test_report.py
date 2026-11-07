from __future__ import annotations

import csv
from pathlib import Path

from recommender.experiments import generate_report


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_report_writes_summary_markdown_and_png(tmp_path: Path) -> None:
    results = tmp_path / "final-results.csv"
    paired = tmp_path / "paired-results.csv"
    slices = tmp_path / "final-slices.csv"
    _write(
        results,
        [
            "artifact_path",
            "ci_high",
            "ci_low",
            "model",
            "ndcg@10",
            "ndcg@10_estimate",
            "ndcg@10_ci_low",
            "ndcg@10_ci_high",
            "run_id",
            "seed",
        ],
        [
            {
                "artifact_path": "model.json",
                "ci_high": 0.8,
                "ci_low": 0.4,
                "model": "popularity",
                "ndcg@10": 0.6,
                "ndcg@10_estimate": 0.6,
                "ndcg@10_ci_low": 0.4,
                "ndcg@10_ci_high": 0.8,
                "run_id": "run-1",
                "seed": 0,
            }
        ],
    )
    _write(
        paired,
        ["baseline_model", "ci_high", "ci_low", "estimate", "leading_model", "metric"],
        [
            {
                "baseline_model": "popularity",
                "ci_high": 0.2,
                "ci_low": -0.1,
                "estimate": 0.05,
                "leading_model": "popularity",
                "metric": "ndcg@10",
            }
        ],
    )
    _write(slices, ["model", "slice"], [{"model": "popularity", "slice": "history_length"}])

    result = generate_report(results, paired, slices, tmp_path / "report")

    assert result.report_path.read_text(encoding="utf-8").startswith("# Final comparison report")
    assert "popularity" in result.summary_path.read_text(encoding="utf-8")
    assert result.figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
