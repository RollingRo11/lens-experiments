"""PCA, t-SNE and UMAP of one layer's 64-dim router projection, in 2D and 3D, with UMAP
tuned for that layer.

UMAP is run on a grid of (n_neighbors, min_dist). Each embedding is scored two ways:
- trustworthiness (k = 15): are a point's neighbours in the embedding also its
  neighbours in 64 dims? Local structure. 1 = perfect.
- distance correlation: Spearman correlation between pairwise distances in 64 dims and
  in the embedding. Global layout (how far apart the groups sit). 1 = perfect.
The UMAP setting with the highest mean of the two is kept, for 2D and 3D. PCA is the
first 2 or 3 coordinates of the router projection (its top principal directions).

Needs only numpy/scipy/scikit-learn/umap-learn (no torch):
    uv run --no-project --python 3.12 --with umap-learn --with scikit-learn \
        python experiments/dimensionality/embed_layer.py --layer 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import umap
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE, trustworthiness

UMAP_NEIGHBORS = [15, 30, 100, 300]
UMAP_MIN_DISTS = [0.1, 0.5]
TRUST_POINTS, TRUST_K = 5000, 15
CORRELATION_POINTS = 2000


def scores(points_SD: np.ndarray, embedded_SE: np.ndarray, rng_seed: int) -> dict:
    rng = np.random.default_rng(rng_seed)
    trust_rows = rng.choice(len(points_SD), TRUST_POINTS, replace=False)
    corr_rows = rng.choice(len(points_SD), CORRELATION_POINTS, replace=False)
    trust = trustworthiness(points_SD[trust_rows], embedded_SE[trust_rows], n_neighbors=TRUST_K)
    corr = spearmanr(pdist(points_SD[corr_rows]), pdist(embedded_SE[corr_rows])).statistic
    return {"trust": float(trust), "dist_corr": float(corr), "mean": float((trust + corr) / 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="results/dimensionality/embed_inputs.npz")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/dimensionality/embed_3d")
    args = parser.parse_args()

    points_SD = np.load(args.inputs)[f"proj_{args.layer}"].astype(np.float64)
    saved: dict[str, np.ndarray] = {}
    table: list[dict] = []

    def record(name: str, embedded_SE: np.ndarray, **params) -> dict:
        row = {"method": name, "dims": embedded_SE.shape[1], **params,
               **scores(points_SD, embedded_SE, args.seed)}
        table.append(row)
        print(json.dumps(row), flush=True)
        return row

    for dims in (2, 3):
        saved[f"pca{dims}"] = points_SD[:, :dims]
        record("pca", saved[f"pca{dims}"])

        start = time.time()
        saved[f"tsne{dims}"] = TSNE(
            n_components=dims, init="pca", random_state=args.seed, n_jobs=-1
        ).fit_transform(points_SD)
        record("tsne", saved[f"tsne{dims}"], perplexity=30, seconds=round(time.time() - start))

    best = None
    for n_neighbors, min_dist in itertools.product(UMAP_NEIGHBORS, UMAP_MIN_DISTS):
        start = time.time()
        embedded_S2 = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=args.seed
        ).fit_transform(points_SD)
        row = record("umap", embedded_S2, n_neighbors=n_neighbors, min_dist=min_dist,
                     seconds=round(time.time() - start))
        if (n_neighbors, min_dist) == (15, 0.1):
            saved["umap2_default"] = embedded_S2
        if best is None or row["mean"] > best[0]["mean"]:
            best = (row, embedded_S2)
    best_row, saved["umap2"] = best
    best_params = {"n_neighbors": best_row["n_neighbors"], "min_dist": best_row["min_dist"]}
    saved["umap3"] = umap.UMAP(n_components=3, random_state=args.seed, **best_params).fit_transform(
        points_SD
    )
    record("umap", saved["umap3"], **best_params)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"layer{args.layer}.npz", **{k: v.astype(np.float32) for k, v in saved.items()})
    (out_dir / f"layer{args.layer}_scores.json").write_text(
        json.dumps({"layer": args.layer, "umap_best": best_params, "table": table}, indent=1)
    )
    print(f"layer {args.layer}: best UMAP {best_params}", flush=True)


if __name__ == "__main__":
    main()
