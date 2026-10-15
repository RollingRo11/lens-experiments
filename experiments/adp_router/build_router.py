"""Build a router from the ADP labels: the 20k labelled points are the reference points and a
new activation takes the label of its nearest one, in the same unit-normalised full space
(identity projection, no PCA). One K for all layers, the largest ADP cluster count; a layer
with fewer clusters leaves the extra experts empty (merge-experts falls them back to the pooled
Jacobian). Runs in the jpp_lens venv.
"""

import json

import numpy as np
import torch as t

from workspace_lens.routing.router import ActivationRouterCollection, LayerActivationRouter

LAYERS = [4, 8, 12, 16, 20, 24, 28]
OUT = "artifacts/gemma4_e2b/router_adp.pt"

points = np.load("results/adp_router/points.npz")
labels = {layer: np.load(f"results/adp_router/labels_layer{layer}.npy") for layer in LAYERS}
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

# Check: each reference point routes back to its own ADP label.
for layer in LAYERS:
    reference_SN = router.layer_routers_L_dict[layer].centroids_ED
    assert t.equal(router.assign(reference_SN, layer), t.from_numpy(labels[layer]).long()), layer

router.save(OUT)
print("K per layer:", json.dumps({layer: int(labels[layer].max() + 1) for layer in LAYERS}))
print(f"saved {OUT} with num_clusters={num_clusters}")
