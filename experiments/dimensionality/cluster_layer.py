"""ADP and HDBSCAN clusterings of one layer's 20k displayed points, in the router's 64-dim
projection (the space the K = 8 router's k-means works in), for a few settings each.

- ADP (Advanced Density Peaks, DADApy): k-nearest-neighbour distances, the TwoNN
  intrinsic dimension, the PAk density, then density-peak clustering. One setting, Z:
  how significant a density peak must be to stay its own cluster (higher Z, fewer
  clusters). Every point gets a cluster (no noise).
- HDBSCAN (scikit-learn): min_cluster_size and cluster_selection_method. Points that fit
  no cluster are labelled -1 (noise).

Needs a numpy 1.x env with DADApy built against it (its C code does not build on numpy 2):
    UV_BUILD_CONSTRAINT=<file with "numpy<2"> uv run --no-project --python 3.12 \
        --with "numpy<2" --with dadapy --with scikit-learn \
        python experiments/dimensionality/cluster_layer.py --layer 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from dadapy import Data
from sklearn.cluster import HDBSCAN

ADP_ZS = [1.65, 3.0, 5.0]
ADP_MAXK = 100
HDBSCAN_MIN_SIZES = [25, 100, 400]
HDBSCAN_SELECTIONS = ["eom", "leaf"]


def summary(labels_S: np.ndarray) -> dict:
    sizes = np.bincount(labels_S[labels_S >= 0]) if (labels_S >= 0).any() else np.array([0])
    return {
        "num_clusters": int((sizes > 0).sum()),
        "noise_fraction": float((labels_S < 0).mean()),
        "largest_fraction": float(sizes.max() / len(labels_S)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="results/dimensionality/embed_inputs.npz")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out-dir", default="results/dimensionality/clusters")
    args = parser.parse_args()

    points_SD = np.load(args.inputs)[f"proj_{args.layer}"].astype(np.float64)
    saved: dict[str, np.ndarray] = {}
    runs: list[dict] = []

    start = time.time()
    data = Data(points_SD, verbose=False)
    data.compute_distances(maxk=ADP_MAXK)
    intrinsic_dim, _, _ = data.compute_id_2NN()
    data.compute_density_PAk()
    density_seconds = round(time.time() - start)
    for z in ADP_ZS:
        start = time.time()
        labels_S = np.asarray(data.compute_clustering_ADP(Z=z, halo=False), dtype=np.int64)
        saved[f"adp_Z{z}"] = labels_S
        runs.append({"algorithm": "adp", "Z": z, **summary(labels_S),
                     "seconds": round(time.time() - start) + density_seconds})
        print(json.dumps(runs[-1]), flush=True)

    for min_size in HDBSCAN_MIN_SIZES:
        for selection in HDBSCAN_SELECTIONS:
            start = time.time()
            labels_S = HDBSCAN(
                min_cluster_size=min_size, cluster_selection_method=selection, n_jobs=-1
            ).fit_predict(points_SD).astype(np.int64)
            saved[f"hdbscan_m{min_size}_{selection}"] = labels_S
            runs.append({"algorithm": "hdbscan", "min_cluster_size": min_size,
                         "selection": selection, **summary(labels_S),
                         "seconds": round(time.time() - start)})
            print(json.dumps(runs[-1]), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"layer{args.layer}.npz", **saved)
    (out_dir / f"layer{args.layer}_runs.json").write_text(json.dumps(
        {"layer": args.layer, "adp_intrinsic_dim": float(intrinsic_dim), "runs": runs}, indent=1
    ))


if __name__ == "__main__":
    main()
