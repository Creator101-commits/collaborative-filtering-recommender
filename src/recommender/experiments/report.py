"""Reproducible tables and a Matplotlib report for final results."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from recommender.experiments.protocol import FINAL_MODEL_NAMES

DEFAULT_REPORT_DIR = Path("artifacts/report")


class ReportError(ValueError):
    """Raised when final result tables cannot produce a report."""


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Paths produced by report generation."""

    report_path: Path
    summary_path: Path
    figure_path: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        raise ReportError(f"cannot read result table {path}: {error}") from error
    if not rows:
        raise ReportError(f"result table {path} is empty")
    return [{key: value or "" for key, value in row.items()} for row in rows]


def _number(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError("result must be finite")
        return value
    except (KeyError, ValueError) as error:
        raise ReportError(f"{path} has invalid numeric field {key}") from error


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ReportError("cannot write an empty report summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _write_figure(path: Path, summary: list[dict[str, object]]) -> None:
    values: dict[str, list[float]] = defaultdict(list)
    for summary_row in summary:
        if summary_row["metric"] == "ndcg@10":
            values[str(summary_row["model"])].append(float(str(summary_row["estimate"])))
    if not values:
        raise ReportError("summary has no ndcg@10 rows for the figure")
    models = [name for name in FINAL_MODEL_NAMES if name in values]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, max(3, len(models) * 0.6)))
    scores = [_mean(values[name]) for name in models]
    bars = ax.barh(models, scores, color="#2f6690")
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="NDCG@10", title="Mean test NDCG@10 across declared seeds")
    ax.bar_label(bars, fmt="%.4f", padding=4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_report(
    results_path: Path,
    paired_results_path: Path,
    slices_path: Path,
    output_dir: Path,
) -> ReportResult:
    """Generate tables, a Matplotlib PNG figure, and Markdown from final CSV artifacts."""
    results = _read_csv(results_path)
    paired = _read_csv(paired_results_path)
    slices = _read_csv(slices_path)
    summary: list[dict[str, object]] = []
    for row in results:
        model = row.get("model", "")
        if model not in FINAL_MODEL_NAMES:
            raise ReportError(f"unknown model in {results_path}: {model}")
        for key in row:
            if not key.endswith("_estimate"):
                continue
            metric = key.removesuffix("_estimate")
            summary.append(
                {
                    "artifact_path": row.get("artifact_path", ""),
                    "ci_high": _number(row, f"{metric}_ci_high", results_path),
                    "ci_low": _number(row, f"{metric}_ci_low", results_path),
                    "estimate": _number(row, f"{metric}_estimate", results_path),
                    "metric": metric,
                    "model": model,
                    "run_id": row.get("run_id", ""),
                    "seed": row.get("seed", ""),
                }
            )
    if not summary:
        raise ReportError(f"no bootstrap metric rows found in {results_path}")
    order = {name: index for index, name in enumerate(FINAL_MODEL_NAMES)}
    summary.sort(key=lambda row: (order[str(row["model"])], str(row["metric"]), str(row["seed"])))
    summary_path = output_dir / "summary.csv"
    figure_path = output_dir / "ndcg-at-10.png"
    report_path = output_dir / "report.md"
    _write_csv(summary_path, summary)
    _write_figure(figure_path, summary)

    ndcg = [row for row in summary if row["metric"] == "ndcg@10"]
    model_values: dict[str, list[float]] = defaultdict(list)
    for summary_row in ndcg:
        model_values[str(summary_row["model"])].append(float(str(summary_row["estimate"])))
    table = [
        "| Model | Mean NDCG@10 | Declared seeds |",
        "|---|---:|---:|",
    ]
    for model in FINAL_MODEL_NAMES:
        if model in model_values:
            table.append(
                f"| {model} | {_mean(model_values[model]):.4f} | {len(model_values[model])} |"
            )
    paired_lines = [
        f"- `{row.get('leading_model', '')}` minus `{row.get('baseline_model', '')}` "
        f"({row.get('metric', '')}): {row.get('estimate', '')} "
        f"[{row.get('ci_low', '')}, {row.get('ci_high', '')}]"
        for row in paired
    ]
    report = "\n".join(
        [
            "# Final comparison report",
            "",
            "This report is generated only from the frozen final CSV artifacts. "
            "It does not retrain models or change their configurations.",
            "",
            "## Primary results",
            "",
            *table,
            "",
            "The figure shows descriptive mean NDCG@10 across the declared learned-model "
            "seeds; it is not a new model-selection step.",
            "",
            f"![Mean NDCG@10](./{figure_path.name})",
            "",
            "## Paired differences",
            "",
            *(paired_lines or ["No paired comparison rows were supplied."]),
            "",
            "## Slice coverage",
            "",
            f"The final slice table contains {len(slices)} rows across the locked "
            "slice definitions.",
            "",
            "## Limitations",
            "",
            "Bootstrap intervals resample users and do not model every dependency caused by "
            "shared items. The leading model is chosen descriptively from these test scores, "
            "so paired intervals against it are exploratory and do not adjust for selection. "
            "Results should be interpreted with the recorded configuration, "
            "manifest hash, run IDs, and Git revision.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return ReportResult(report_path, summary_path, figure_path)
