"""ADP cluster count vs Z on one layer's 20k raw points (same setup as adp_layer.py): the
density is computed once, then ADP runs for each Z. Saves the labels for every Z."""

import argparse
import json

import numpy as np
from dadapy import Data

ZS = [1.65, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

data = Data(np.load("results/adp_router/points.npz")[f"raw_{args.layer}"].astype(np.float64), verbose=False)
data.compute_distances(maxk=100)
data.compute_id_2NN()
data.compute_density_PAk()
rows = []
for z in ZS:
    labels_S = np.asarray(data.compute_clustering_ADP(Z=z, halo=False), dtype=np.int64)
    np.save(f"results/adp_router/zsweep/labels_layer{args.layer}_Z{z}.npy", labels_S)
    sizes = np.bincount(labels_S[labels_S >= 0])
    rows.append({"Z": z, "num_clusters": int(len(sizes)), "unlabelled": int((labels_S < 0).sum()),
                 "largest_fraction": round(float(sizes.max() / len(labels_S)), 3)})
json.dump(rows, open(f"results/adp_router/zsweep/layer{args.layer}.json", "w"), indent=1)
print(args.layer, json.dumps(rows), flush=True)
