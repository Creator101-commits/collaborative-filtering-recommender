"""Reproduce image.png from actual code outputs; no invented benchmark values.

Requires the project dependencies and an environment with Matplotlib installed.
Run from the repository root with PYTHONPATH=src python scripts/render_figure.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from recommender.evaluation.metrics import ndcg_at_k, recall_at_k
from recommender.models import (
    BiasedMatrixFactorizationRecommender,
    BPRRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    UserKNNRecommender,
)

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b"]
ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")


pairs = [
    (0, 0, 5),
    (0, 1, 4),
    (0, 2, 1),
    (1, 1, 5),
    (1, 2, 4),
    (1, 3, 2),
    (2, 2, 5),
    (2, 3, 4),
    (2, 4, 1),
    (3, 3, 5),
    (3, 4, 4),
    (3, 5, 2),
]
rows = [dict(user_idx=u, item_idx=i, rating=r) for u, i, r in pairs]
held_out = {0: {3}, 1: {4}, 2: {5}, 3: {0}}
matrix = np.full((4, 6), np.nan)
for u, i, r in pairs:
    matrix[u, i] = r
image = axes[0].imshow(matrix, vmin=1, vmax=5, cmap="Blues", aspect="auto")
axes[0].set(
    xticks=range(6),
    yticks=range(4),
    xlabel="Item index",
    ylabel="User index",
    title="Training ratings (12 fixture rows)",
)
fig.colorbar(image, ax=axes[0], shrink=0.75, label="Rating; blank = unseen")
models = {
    "Popularity": PopularityRecommender(6),
    "User KNN": UserKNNRecommender(4, 6),
    "Item KNN": ItemKNNRecommender(4, 6),
    "MF": BiasedMatrixFactorizationRecommender(4, 6, epochs=20, latent_dim=4),
    "BPR": BPRRecommender(4, 6, epochs=20, latent_dim=4),
}
ndcg, recall, scores = [], [], []
for name, model in models.items():
    model.fit(rows, seed=42)
    recommendations = {}
    for u in range(4):
        seen = {i for user, i, _ in pairs if user == u}
        recommendations[u] = list(model.recommend(u, set(range(6)) - seen, 2))
    ndcg.append(float(np.mean([ndcg_at_k(recommendations[u], held_out[u], 2) for u in range(4)])))
    recall.append(
        float(np.mean([recall_at_k(recommendations[u], held_out[u], 2) for u in range(4)]))
    )
    scores.append([model.score(0, i) for i in (3, 4, 5)])
x = np.arange(len(models))
axes[1].bar(x - 0.18, ndcg, 0.36, label="NDCG@2", color=COLORS[0])
axes[1].bar(x + 0.18, recall, 0.36, label="Recall@2", color=COLORS[1])
axes[1].set(
    xticks=x, xticklabels=list(models), ylim=(0, 1.12), title="Measured tiny-fixture ranking"
)
axes[1].tick_params(axis="x", rotation=25)
axes[1].legend(fontsize=9)
# Scores from the same model share a scale; cross-model scores do not.
axes[2].bar(["Item 3", "Item 4", "Item 5"], scores[2], color=COLORS[2])
axes[2].set(title="Item KNN: user 0 unseen items", ylabel="Weighted rating score")
fig.suptitle("Collaborative filtering | reproducible four-user demonstration", fontsize=16)
fig.supxlabel(
    "Synthetic interactions, six-item catalog, seed 42. These are not MovieLens 1M "
    "benchmark results.",
    fontsize=10,
)
payload = {
    "source": "explicit synthetic 4-user, 6-item fixture",
    "training": rows,
    "held_out": {str(u): sorted(v) for u, v in held_out.items()},
    "models": list(models),
    "ndcg_at_2": ndcg,
    "recall_at_2": recall,
    "user_0_unseen_scores": scores,
}

source_hash = hashlib.sha256()
for source_path in sorted((ROOT / "src").rglob("*.py")):
    source_hash.update(str(source_path.relative_to(ROOT)).encode())
    source_hash.update(source_path.read_bytes())
payload["source_sha256"] = source_hash.hexdigest()

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
plt.close(fig)
(ROOT / "scripts" / "figure-data.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
