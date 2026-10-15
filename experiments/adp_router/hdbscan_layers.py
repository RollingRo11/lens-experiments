"""HDBSCAN on each layer's 20k unit-normalised raw (1536-dim) points, the same points ADP used.
Sweeps min_cluster_size and keeps, per layer, the setting whose cluster count is closest to 8
(ties: less noise). Noise points keep label -1; the router passes that on, so tokens nearest a
noise point are dropped from every expert.

Needs scikit-learn (run in the uv env from slurm_hdbscan.sh).
"""

import argparse
import json
import time

import numpy as np
from sklearn.cluster import HDBSCAN

LAYERS = [4, 8, 12, 16, 20, 24, 28]
MIN_CLUSTER_SIZES = [100, 200, 400, 800]
MIN_SAMPLES = 10
TARGET = 8

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, default=None, help="One layer (default: all).")
args = parser.parse_args()

points = np.load("results/adp_router/points.npz")
summary = {}
for layer in [args.layer] if args.layer is not None else LAYERS:
    points_SN = points[f"raw_{layer}"].astype(np.float64)
    runs = []
    for size in MIN_CLUSTER_SIZES:
        start = time.time()
        labels_S = HDBSCAN(min_cluster_size=size, min_samples=MIN_SAMPLES).fit_predict(points_SN)
        sizes = np.bincount(labels_S[labels_S >= 0]) if (labels_S >= 0).any() else np.array([])
        runs.append({"min_cluster_size": size, "num_clusters": int(len(sizes)),
                     "noise_fraction": round(float((labels_S < 0).mean()), 3),
                     "largest_fraction": round(float(sizes.max() / len(labels_S)), 3) if len(sizes) else 0.0,
                     "seconds": round(time.time() - start)})
        np.save(f"results/adp_router/hdbscan/labels_layer{layer}_mcs{size}.npy", labels_S)
        print(layer, json.dumps(runs[-1]), flush=True)
    best = min((r for r in runs if r["num_clusters"] > 0),
               key=lambda r: (abs(r["num_clusters"] - TARGET), r["noise_fraction"]))
    summary[layer] = {"chosen": best, "runs": runs}
    labels_S = np.load(f"results/adp_router/hdbscan/labels_layer{layer}_mcs{best['min_cluster_size']}.npy")
    np.save(f"results/adp_router/hdbscan/labels_layer{layer}.npy", labels_S)
    print("CHOSEN", layer, json.dumps(best), flush=True)
name = f"summary_layer{args.layer}" if args.layer is not None else "summary"
json.dump(summary, open(f"results/adp_router/hdbscan/{name}.json", "w"), indent=1)
