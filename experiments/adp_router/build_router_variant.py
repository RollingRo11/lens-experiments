"""Build a nearest-labelled-point router from saved per-layer labels (see build_router.py).

--labels: path pattern with {layer}. --shuffle: random labels with the same cluster sizes (a control). --top N: keep the N largest clusters and put every other
point in one extra cluster (N + 1 experts). Label -1 (HDBSCAN noise) stays -1: tokens routed to
it join no expert, so they are dropped from the fit.
"""

import argparse
import json

import numpy as np
import torch as t

from workspace_lens.routing.router import ActivationRouterCollection, LayerActivationRouter

LAYERS = [4, 8, 12, 16, 20, 24, 28]

parser = argparse.ArgumentParser()
parser.add_argument("--labels", required=True)
parser.add_argument("--top", type=int, default=None)
parser.add_argument("--out", required=True)
parser.add_argument(
    "--shuffle", action="store_true",
    help="Control: shuffle the labels across the 20k points (seed 0), keeping cluster sizes. "
    "Tokens then land in experts at random, so the experts are no longer specialised.",
)
args = parser.parse_args()

points = np.load("results/adp_router/points.npz")
labels = {}
for layer in LAYERS:
    labels_S = np.load(args.labels.format(layer=layer)).astype(np.int64)
    if args.shuffle:
        labels_S = np.random.default_rng(0).permutation(labels_S)
    # Relabel clusters by size, largest first (noise stays -1).
    kept = labels_S >= 0
    order = np.argsort(-np.bincount(labels_S[kept]), kind="stable")
    rank_of_label = np.empty_like(order)
    rank_of_label[order] = np.arange(len(order))
    labels_S[kept] = rank_of_label[labels_S[kept]]
    if args.top is not None:
        labels_S[labels_S >= args.top] = args.top
    labels[layer] = labels_S

num_clusters = int(max(labels_S.max() + 1 for labels_S in labels.values()))
d_model = points[f"raw_{LAYERS[0]}"].shape[1]
router = ActivationRouterCollection(num_clusters=num_clusters, projection_dim=d_model)
for layer in LAYERS:
    router.layer_routers_L_dict[layer] = LayerActivationRouter(
        pca_mean_N=t.zeros(d_model),
        pca_components_ND=t.eye(d_model),
        centroids_ED=t.from_numpy(points[f"raw_{layer}"]).float(),
        centroid_labels_C=t.from_numpy(labels[layer]).long(),
    )
    reference_SN = router.layer_routers_L_dict[layer].centroids_ED
    assert t.equal(router.assign(reference_SN, layer), t.from_numpy(labels[layer]).long()), layer

router.save(args.out)
for layer in LAYERS:
    labels_S = labels[layer]
    sizes = np.bincount(labels_S[labels_S >= 0], minlength=num_clusters)
    print(layer, json.dumps({"sizes": sizes.tolist(), "noise": int((labels_S < 0).sum())}))
print(f"saved {args.out} with num_clusters={num_clusters}")
