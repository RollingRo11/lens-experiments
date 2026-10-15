"""t-SNE and UMAP of the router's 64-dim projection, down to 2 dims, per layer.

The same random subset of rows is used for every layer and both methods. Each point
also carries its cluster under the saved K=8 router (nearest centroid), for colouring.
Library defaults throughout (t-SNE: perplexity 30, PCA init; UMAP: 15 neighbours,
min_dist 0.1).

    uv run --with umap-learn --with scikit-learn python experiments/dimensionality/embed_2d.py
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch as t
import umap
from sklearn.manifold import TSNE

from workspace_lens.routing.router import ActivationRouterCollection

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proj-dir", default="results/dimensionality/proj64")
    parser.add_argument("--router-path", default="results/dimensionality/proj64/router.pt")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28])
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/dimensionality/embed_2d.npz")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    router = ActivationRouterCollection.load(args.router_path)
    index = t.load(Path(args.proj_dir) / "index.pt", weights_only=True)
    total_points = index["position"].shape[0]
    rows_Int_S = t.randperm(total_points, generator=t.Generator().manual_seed(args.seed))[
        : args.num_points
    ].sort().values

    saved: dict[str, np.ndarray] = {
        "rows": rows_Int_S.numpy(),
        "position": index["position"][rows_Int_S].numpy(),
        "token_id": index["token_id"][rows_Int_S].numpy(),
    }
    for layer in args.layers:
        projected_SD = t.load(
            Path(args.proj_dir) / f"proj{router.projection_dim}_layer{layer}.pt", weights_only=True
        )[rows_Int_S]
        centroids_ED = router.layer_routers_L_dict[layer].centroids_ED
        saved[f"cluster_{layer}"] = t.cdist(projected_SD, centroids_ED).argmin(dim=1).numpy()

        points_SD = projected_SD.numpy()
        start = time.time()
        saved[f"tsne_{layer}"] = TSNE(
            n_components=2, init="pca", random_state=args.seed, n_jobs=-1
        ).fit_transform(points_SD)
        logger.info("layer %d: t-SNE %.0fs", layer, time.time() - start)
        start = time.time()
        saved[f"umap_{layer}"] = umap.UMAP(
            n_components=2, random_state=args.seed
        ).fit_transform(points_SD)
        logger.info("layer %d: UMAP %.0fs", layer, time.time() - start)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **saved)
    logger.info("saved %s", out_path)


if __name__ == "__main__":
    main()
