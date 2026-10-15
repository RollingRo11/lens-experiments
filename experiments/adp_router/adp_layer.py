"""ADP (DADApy) on one layer's 20k unit-normalised activations in the full d_model space:
k-nearest-neighbour distances, the TwoNN intrinsic dimension, the PAk density (which uses that
dimension), then density-peak clustering at one Z. Every point gets a cluster (no noise).

Needs the numpy 1.x env (see experiments/dimensionality/slurm_cluster_layers.sh).
"""

import argparse
import json
import time

import numpy as np
from dadapy import Data

MAXK = 100

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--z", type=float, default=1.65)
parser.add_argument("--points", default="results/adp_router/points.npz")
args = parser.parse_args()

points_SN = np.load(args.points)[f"raw_{args.layer}"].astype(np.float64)
start = time.time()
data = Data(points_SN, verbose=False)
data.compute_distances(maxk=MAXK)
intrinsic_dim, intrinsic_dim_err, _ = data.compute_id_2NN()
data.compute_density_PAk()
labels_S = np.asarray(data.compute_clustering_ADP(Z=args.z, halo=False), dtype=np.int64)
sizes = np.bincount(labels_S)
summary = {
    "layer": args.layer,
    "Z": args.z,
    "num_points": len(labels_S),
    "twonn_id": float(intrinsic_dim),
    "twonn_id_err": float(intrinsic_dim_err),
    "num_clusters": int(len(sizes)),
    "largest_fraction": float(sizes.max() / len(labels_S)),
    "sizes": sorted(sizes.tolist(), reverse=True),
    "seconds": round(time.time() - start),
}
np.save(f"results/adp_router/labels_layer{args.layer}.npy", labels_S)
with open(f"results/adp_router/summary_layer{args.layer}.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "sizes"}), flush=True)
