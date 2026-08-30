"""Reproducible tables and a dependency-free report for final results."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
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
        return float(row[key])
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
            values[str(summary_row["model"])].append(
                float(str(summary_row["estimate"]))
            )
    if not values:
        raise ReportError("summary has no ndcg@10 rows for the figure")
    models = [name for name in FINAL_MODEL_NAMES if name in values]
    width, height = 800, 90 + 45 * len(models)
    chart_width = 560
    bars = []
    for index, model in enumerate(models):
        value = _mean(values[model])
        y = 45 + index * 45
        bars.append(
            f'<text x="10" y="{y + 16}" font-size="13">{escape(model)}</text>'
            f'<rect x="220" y="{y}" width="{value * chart_width:.2f}" height="24" fill="#2f6690" />'
            f'<text x="{230 + value * chart_width:.2f}" y="{y + 17}" '
            f'font-size="12">{value:.4f}</text>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white" />',
                '<text x="10" y="25" font-size="16" font-weight="bold">'
                "Mean test NDCG@10 by model and declared seed</text>",
                *bars,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def generate_report(
    results_path: Path,
    paired_results_path: Path,
    slices_path: Path,
    output_dir: Path,
) -> ReportResult:
    """Generate tables, an SVG figure, and Markdown from final CSV artifacts."""
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
                    "estimate": _number(row, metric, results_path),
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
    figure_path = output_dir / "ndcg-at-10.svg"
    report_path = output_dir / "report.md"
    _write_csv(summary_path, summary)
    _write_figure(figure_path, summary)

    ndcg = [row for row in summary if row["metric"] == "ndcg@10"]
    model_values: dict[str, list[float]] = defaultdict(list)
    for summary_row in ndcg:
        model_values[str(summary_row["model"])].append(
            float(str(summary_row["estimate"]))
        )
    table = [
        "| Model | Mean NDCG@10 | Declared seeds |",
        "|---|---:|---:|",
    ]
    for model in FINAL_MODEL_NAMES:
        if model in model_values:
            table.append(
                f"| {model} | {_mean(model_values[model]):.4f} | "
                f"{len(model_values[model])} |"
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
            "shared items. Results should be interpreted with the recorded configuration, "
            "manifest hash, run IDs, and Git revision.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return ReportResult(report_path, summary_path, figure_path)
