"""Routers for the partition controls of fix 4, all bucketing the same fit pass:

- ``random{K}_s{seed}``: every fit position of the 1,024 fitting prompts put in one of K
  buckets at random, sizes equal to within one, independently per layer
  (``random_position_partition``), for K = 8 and 32 and seeds 0, 1, 2.
- ``kmeans20k_K{K}``: k-means (k-means++ start, seed 0) on the same 20k unit-normalised
  full-width (1,536-dim) points ADP clustered (``results/adp_router/points.npz``); a position
  goes to its nearest centroid, for K = 8 and 32.
- ``pooled_K1``: one bucket (the pooled Jacobian, the R-Lens of this pass).

Writes ``<out-dir>/router_<name>.pt`` and prints the bucket sizes. The prompts are tokenised
by the model wrapper the fit uses, so the random partitions find every prompt. Runs in the
jpp_lens venv from the repo root (the model is loaded for its tokenizer wrapper; the GPU is
used for k-means when there is one).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.routing.cluster import kmeans  # noqa: E402
from workspace_lens.routing.router import (  # noqa: E402
    ActivationRouterCollection,
    LayerActivationRouter,
    random_position_partition,
)

LAYERS = [4, 8, 12, 16, 20, 24, 28]


def identity_router(centroids_L_dict_KN: dict[int, t.Tensor]) -> ActivationRouterCollection:
    """Nearest centroid among the given ones, after unit-normalising (no PCA)."""
    d_model = next(iter(centroids_L_dict_KN.values())).shape[1]
    num_clusters = next(iter(centroids_L_dict_KN.values())).shape[0]
    router = ActivationRouterCollection(num_clusters=num_clusters, projection_dim=d_model)
    for layer, centroids_KN in centroids_L_dict_KN.items():
        router.layer_routers_L_dict[layer] = LayerActivationRouter(
            pca_mean_N=t.zeros(d_model), pca_components_ND=t.eye(d_model), centroids_ED=centroids_KN.cpu().float()
        )
    return router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=jpp_cli.RECIPE_MAX_SEQ_LEN)
    parser.add_argument("--skip-first-n-positions", type=int, default=jpp_cli.RECIPE_SKIP_FIRST_N_POSITIONS)
    parser.add_argument("--points", default="results/adp_router/points.npz")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if t.cuda.is_available() else "cpu"

    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    prompts = jpp_cli.load_fit_prompts(args.num_prompts)
    input_ids_list_S = [model.encode(prompt, max_length=args.max_seq_len)[0].cpu() for prompt in prompts]
    for num_clusters in (8, 32):
        for seed in (0, 1, 2):
            router = random_position_partition(
                input_ids_list_S, num_clusters=num_clusters, layers=LAYERS,
                skip_first_n_positions=args.skip_first_n_positions, seed=seed,
            )
            name = f"random{num_clusters}_s{seed}"
            router.save(str(out_dir / f"router_{name}.pt"))
            labels_PS = router.labels_L_dict_PS[LAYERS[0]]
            sizes = t.bincount(labels_PS[labels_PS >= 0], minlength=num_clusters)
            print(name, json.dumps({"prompts": len(router.prompt_hashes), "positions": int(sizes.sum()),
                                    "smallest": int(sizes.min()), "largest": int(sizes.max())}), flush=True)

    points = np.load(args.points)
    for num_clusters in (8, 32):
        centroids_L_dict_KN = {}
        for layer in LAYERS:
            points_SN = t.from_numpy(points[f"raw_{layer}"]).float().to(device)
            centroids_KN, assignments_S = kmeans(points_SN, num_clusters, seed=0)
            centroids_L_dict_KN[layer] = centroids_KN
            sizes = t.bincount(assignments_S, minlength=num_clusters).cpu()
            print(f"kmeans20k_K{num_clusters} layer {layer}", json.dumps(
                {"largest_fraction": round(sizes.max().item() / len(points_SN), 3), "smallest": sizes.min().item()}
            ), flush=True)
        identity_router(centroids_L_dict_KN).save(str(out_dir / f"router_kmeans20k_K{num_clusters}.pt"))

    d_model = points[f"raw_{LAYERS[0]}"].shape[1]
    identity_router({layer: t.zeros(1, d_model) for layer in LAYERS}).save(str(out_dir / "router_pooled_K1.pt"))
    print("wrote routers to", out_dir, flush=True)


if __name__ == "__main__":
    main()
