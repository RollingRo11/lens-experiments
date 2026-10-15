"""Pick 20k of the router's saved activations per layer (seed 0) and write them unit-normalised,
in the full d_model space (no PCA), for ADP. Runs in the jpp_lens venv.

    .venv/bin/python -u experiments/adp_router/prep_points.py
"""

import numpy as np
import torch as t

from workspace_lens.routing.cluster import unit_normalize

ACTIVATIONS_DIR = "artifacts/gemma4_e2b/activations"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
NUM_POINTS = 20_000
OUT = "results/adp_router/points.npz"

arrays = {}
for layer in LAYERS:
    activations_PN = t.load(f"{ACTIVATIONS_DIR}/layer{layer}.pt", weights_only=True)
    rows_S = t.randperm(activations_PN.shape[0], generator=t.Generator().manual_seed(0))[:NUM_POINTS]
    arrays[f"raw_{layer}"] = unit_normalize(activations_PN[rows_S].float()).numpy()
    arrays[f"rows_{layer}"] = rows_S.numpy()
    print(layer, tuple(activations_PN.shape), "->", arrays[f"raw_{layer}"].shape, flush=True)
np.savez(OUT, **arrays)
print("wrote", OUT)
