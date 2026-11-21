"""Reproduce image.png from curated results/ benchmark outputs.

Reads ``results/final-results.csv``, ``results/validation-results.csv``,
and ``results/manifest.json`` and renders three panels: frozen-test
recall@10 and ndcg@10 with bootstrap intervals per model (per-seed spread
for the stochastic models), the validation KNN aggregation comparison,
and the test popularity-bias profile. Writes ``figure-data.json`` with
the plotted values so every number traces to results/.

Requires the project dependencies and an environment with Matplotlib,
e.g. ``uv run --with matplotlib python scripts/render_figure.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODEL_ORDER = ["popularity", "user_knn", "item_knn", "matrix_factorization", "bpr"]
MODEL_LABELS = {
    "popularity": "Popularity",
    "user_knn": "User KNN",
    "item_knn": "Item KNN",
    "matrix_factorization": "MF",
    "bpr": "BPR",
}
MODEL_COLORS = {
    "popularity": "#64748b",
    "user_knn": "#4169e1",
    "item_knn": "#10a68b",
    "matrix_factorization": "#d95f45",
    "bpr": "#8659cf",
}
AGG_COLORS = {"weighted_mean": "#94a3b8", "weighted_sum": "#d95f45"}


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(value: str) -> float:
    return float(value)


def main() -> None:
    final = _rows("final-results.csv")
    validation = _rows("validation-results.csv")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")

    # Panel 1: test recall@10 (solid) and ndcg@10 (hollow) with CIs.
    ax = axes[0]
    xticks = np.arange(len(MODEL_ORDER))
    metric_specs = [
        ("recall@10", "recall@10_estimate", "recall@10_ci_low", "recall@10_ci_high", True),
        ("ndcg@10", "ndcg@10_estimate", "ndcg@10_ci_low", "ndcg@10_ci_high", False),
    ]
    panel1_payload: dict[str, Any] = {}
    offset = {"recall@10": -0.16, "ndcg@10": 0.16}
    for metric, est_col, lo_col, hi_col, solid in metric_specs:
        for i, model in enumerate(MODEL_ORDER):
            seeds = [r for r in final if r["model"] == model]
            ests = np.array([_f(r[est_col]) for r in seeds])
            mean_est = float(ests.mean())
            # Mean CI bounds across seeds for the error bar.
            lo = float(np.mean([_f(r[lo_col]) for r in seeds]))
            hi = float(np.mean([_f(r[hi_col]) for r in seeds]))
            x = i + offset[metric]
            color = MODEL_COLORS[model]
            if solid:
                ax.errorbar(
                    x, mean_est, yerr=[[mean_est - lo], [hi - mean_est]],
                    fmt="o", ms=6, color=color, ecolor=color, elinewidth=1.4,
                    capsize=3, alpha=0.9,
                )
            else:
                ax.errorbar(
                    x, mean_est, yerr=[[mean_est - lo], [hi - mean_est]],
                    fmt="o", ms=6, mfc="white", mec=color, color=color,
                    ecolor=color, elinewidth=1.2, capsize=3, alpha=0.9,
                )
            # Per-seed spread for stochastic models.
            if len(seeds) > 1:
                ax.plot(
                    np.full(len(seeds), x) + np.linspace(-0.05, 0.05, len(seeds)),
                    ests, "|", color=color, ms=8, mew=1.2, alpha=0.8,
                )
            panel1_payload.setdefault(model, {})[metric] = {
                "seeds": [float(r[est_col]) for r in seeds],
                "mean": mean_est,
                "ci_low_mean": lo,
                "ci_high_mean": hi,
            }
    ax.set_xticks(xticks, [MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Frozen test metrics (95% bootstrap CI)")
    ax.set_ylim(0, 0.13)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#334155", ls="",
                   label="recall@10 (solid)"),
            Line2D([0], [0], marker="o", mfc="white", mec="#334155", ls="",
                   label="ndcg@10 (hollow)"),
            Line2D([0], [0], marker="|", color="#334155", ls="",
                   label="per-seed value"),
        ],
        fontsize=8, loc="upper right",
    )

    # Panel 2: validation KNN aggregation comparison (recall@10, log y).
    ax = axes[1]
    knn_rows = [r for r in validation if r["model"] in ("user_knn", "item_knn")]
    groups = []
    for model in ("user_knn", "item_knn"):
        for sim in ("cosine", "adjusted_cosine"):
            groups.append((model, sim))
    gpos = np.arange(len(groups))
    agg_offset = {"weighted_mean": -0.18, "weighted_sum": 0.18}
    panel2_payload: dict[str, Any] = {}
    for agg in ("weighted_mean", "weighted_sum"):
        for j, (model, sim) in enumerate(groups):
            match = [
                r for r in knn_rows
                if r["model"] == model
                and json.loads(r["config"]).get("similarity") == sim
                and json.loads(r["config"]).get("aggregation") == agg
            ]
            if not match:
                continue
            r = match[0]
            val = _f(r["recall@10"])
            x = gpos[j] + agg_offset[agg]
            ax.plot(x, val, "o", ms=8, color=AGG_COLORS[agg])
            ax.annotate(f"{val:.3f}", (x, val), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7.5)
            panel2_payload[f"{model}/{sim}/{agg}"] = {
                "recall@10": val,
                "ndcg@10": _f(r["ndcg@10"]),
                "mean_popularity@10": _f(r["mean_popularity@10"]),
            }
    ax.set_yscale("log")
    ax.set_xticks(gpos, [f"{MODEL_LABELS[m]}\n{sim.replace('_', ' ')}"
                         for m, sim in groups], fontsize=8)
    ax.set_ylabel("Validation recall@10 (log scale)")
    ax.set_title("KNN aggregation: weighted_mean vs weighted_sum")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=AGG_COLORS["weighted_sum"], ls="",
                   label="weighted_sum"),
            Line2D([0], [0], marker="o", color=AGG_COLORS["weighted_mean"], ls="",
                   label="weighted_mean"),
        ],
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
    )

    # Panel 3: popularity bias, mean_popularity@10 per model on test.
    ax = axes[2]
    pops = []
    for model in MODEL_ORDER:
        seeds = [r for r in final if r["model"] == model]
        pops.append(float(np.mean([_f(r["mean_popularity@10"]) for r in seeds])))
    ax.bar(
        xticks, pops,
        color=[MODEL_COLORS[m] for m in MODEL_ORDER], width=0.62,
    )
    for i, v in enumerate(pops):
        ax.annotate(f"{v:,.0f}", (i, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(xticks, [MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("mean_popularity@10")
    ax.set_title("Popularity of recommended items (test)")
    ax.set_ylim(0, max(pops) * 1.18)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Collaborative filtering benchmark | frozen test and validation results",
        fontsize=15,
    )
    fig.supxlabel(
        f"Source: results/final-results.csv and results/validation-results.csv, "
        f"git_sha {manifest.get('git_sha', 'unknown')}, "
        f"protocol {manifest.get('protocol_version', 'unknown')}.",
        fontsize=9,
    )

    payload = {
        "source": "results/ CSVs copied unedited from the Kaggle benchmark run",
        "git_sha": manifest.get("git_sha"),
        "protocol_version": manifest.get("protocol_version"),
        "test_metrics": panel1_payload,
        "knn_validation": panel2_payload,
        "mean_popularity_at_10": {
            MODEL_LABELS[m]: pops[i] for i, m in enumerate(MODEL_ORDER)
        },
    }
    source_hash = hashlib.sha256()
    for name in (
        "final-results.csv", "validation-results.csv", "manifest.json",
        "paired-results.csv",
    ):
        path = RESULTS / name
        if path.exists():
            source_hash.update(name.encode())
            source_hash.update(path.read_bytes())
    payload["results_sha256"] = source_hash.hexdigest()

    fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    (ROOT / "scripts" / "figure-data.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {ROOT / 'image.png'} and scripts/figure-data.json")


if __name__ == "__main__":
    main()
