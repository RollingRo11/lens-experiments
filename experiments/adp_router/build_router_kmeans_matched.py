"""k-means with ADP's cluster count on every layer: same 20k unit-normalised raw (1536-dim)
points as ADP, K_layer = the number of ADP clusters at that layer (Z = 1.65), routed by nearest
centroid. Tests whether the 191-expert ADP lens wins because of ADP's partition or just
because it has more clusters. Runs in the jpp_lens venv (GPU if present).
"""

import argparse
import json

import numpy as np
import torch as t

from workspace_lens.routing.cluster import kmeans
from workspace_lens.routing.router import ActivationRouterCollection, LayerActivationRouter

LAYERS = [4, 8, 12, 16, 20, 24, 28]

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

device = "cuda" if t.cuda.is_available() else "cpu"
points = np.load("results/adp_router/points.npz")
ks = {layer: int(np.load(f"results/adp_router/labels_layer{layer}.npy").max() + 1) for layer in LAYERS}
d_model = points[f"raw_{LAYERS[0]}"].shape[1]
router = ActivationRouterCollection(num_clusters=max(ks.values()), projection_dim=d_model)
for layer in LAYERS:
    points_SN = t.from_numpy(points[f"raw_{layer}"]).float().to(device)
    centroids_KN, assignments_S = kmeans(points_SN, ks[layer], seed=0)
    router.layer_routers_L_dict[layer] = LayerActivationRouter(
        pca_mean_N=t.zeros(d_model), pca_components_ND=t.eye(d_model), centroids_ED=centroids_KN.cpu()
    )
    sizes = t.bincount(assignments_S, minlength=ks[layer]).cpu()
    print(layer, json.dumps({"K": ks[layer], "largest_fraction": round(sizes.max().item() / len(points_SN), 3),
                             "smallest": sizes.min().item()}), flush=True)
router.save(args.out)
print(f"saved {args.out} with num_clusters={router.num_clusters}")
