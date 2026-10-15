"""Routed lenses: instead of one fixed weighted mix of the expert Jacobians, each activation
uses only the experts of the clusters it lives near.

For an activation h at a layer: project it with the fit's router (unit-normalise, then that
router's PCA or identity), find its k nearest reference points (the router's 20k labelled
points, or for the k-means router 20k saved activations labelled by router.assign), and weight
each expert by the share of those k neighbours in its cluster. The readout is
sum_e w_e (J_e h). k = 1 is hard routing (h's own cluster only). Neighbours labelled -1
(HDBSCAN noise) count toward the pooled Jacobian.

Scores the routed lenses next to the J++ (weighted) lenses on the same held-out items.
Runs in the jpp_lens venv from the repo root.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.condense_experts import ExpertJacobians  # noqa: E402
from workspace_lens.routing.router import ActivationRouterCollection  # noqa: E402

OUT = "artifacts/gemma4_e2b"
MODEL = "google/gemma-4-E2B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
CSV = f"{OUT}/model_correctness_stripped.csv"
POINTS = "results/adp_router/points.npz"


class RoutedLens:
    """Duck-typed lens (what ReadoutEvalRunner uses: config, source_layers,
    move_parameters_to_device, transport)."""

    def __init__(self, experts: ExpertJacobians, router: ActivationRouterCollection, k: int):
        self.config = experts.config.as_jacobian(checkpoint_name=f"routed_k{k}")
        self.source_layers = experts.layers
        self.k = k
        self.router = router
        self.experts_L_dict_EFN = experts.experts_L_dict_EFN
        self.pooled_L_dict_FN = experts.pooled_jacobian_L_dict_FN
        points = np.load(POINTS)
        self.refs_L_dict_SD: dict[int, t.Tensor] = {}
        self.labels_L_dict_S: dict[int, t.Tensor] = {}
        for layer in self.source_layers:
            raw_SN = t.from_numpy(points[f"raw_{layer}"]).float()
            layer_router = router.layer_routers_L_dict[layer]
            if layer_router.centroid_labels_C is not None:
                # Labelled router: its reference points are the same 20k points.
                self.refs_L_dict_SD[layer] = layer_router.centroids_ED.float()
                self.labels_L_dict_S[layer] = layer_router.centroid_labels_C.long()
            else:
                self.refs_L_dict_SD[layer] = layer_router.project(raw_SN)
                self.labels_L_dict_S[layer] = layer_router.assign(raw_SN)

    def move_parameters_to_device(self, device) -> None:
        self.device = device
        for d in (self.refs_L_dict_SD, self.labels_L_dict_S, self.pooled_L_dict_FN):
            for layer in d:
                d[layer] = d[layer].to(device)
        # Experts stay on the CPU (191-expert files are ~1 GB per layer); used ones are moved per call.

    def transport(self, residual_PN: t.Tensor, layer: int) -> t.Tensor:
        shape = residual_PN.shape
        flat_PN = residual_PN.reshape(-1, shape[-1]).float()
        layer_router = self.router.layer_routers_L_dict[layer]
        query_PD = layer_router.project(flat_PN)
        neighbours_PK = t.cdist(query_PD, self.refs_L_dict_SD[layer]).topk(self.k, largest=False).indices
        neighbour_labels_PK = self.labels_L_dict_S[layer][neighbours_PK]
        out_PF = t.zeros_like(flat_PN)
        for label in neighbour_labels_PK.unique().tolist():
            weight_P = (neighbour_labels_PK == label).float().mean(dim=1)
            jacobian_FN = (
                self.pooled_L_dict_FN[layer] if label < 0
                else self.experts_L_dict_EFN[layer][label].to(flat_PN.device)
            ).float()
            out_PF += weight_P[:, None] * (flat_PN @ jacobian_FN.T)
        return out_PF.reshape(shape).to(residual_PN.dtype)


parser = argparse.ArgumentParser()
parser.add_argument("--variants", nargs="+", default=["v2", "adp", "adp_top7", "hdbscan"])
parser.add_argument("--ks", type=int, nargs="+", default=[1, 20])
parser.add_argument("--out", default=f"{OUT}/eval_held_out_routed.csv")
parser.add_argument("--extra-lenses", nargs="*", default=[], help="More lens files (or 'logit') to score alongside.")
args = parser.parse_args()

# variant -> (experts file, router file, J++ lens file)
FILES = {
    "v2": ("experts.pt", "router.pt", "jpp_lens_v2.pt"),
    "adp": ("experts_adp.pt", "router_adp.pt", "jpp_lens_adp.pt"),
    "adp_top7": ("experts_adp_top7.pt", "router_adp_top7.pt", "jpp_lens_adp_top7.pt"),
    "hdbscan": ("experts_hdbscan.pt", "router_hdbscan.pt", "jpp_lens_hdbscan.pt"),
}

model = get_hf_model(MODEL, attn_implementation="sdpa")
lenses = jpp_cli.load_lenses(
    [f"{OUT}/{FILES[v][2]}" for v in args.variants] + [f"{OUT}/pooled_lens.pt"] + args.extra_lenses,
    model, hf_model_name=MODEL, layers=LAYERS,
)
for variant in args.variants:
    experts = ExpertJacobians.load(f"{OUT}/{FILES[variant][0]}")
    router = ActivationRouterCollection.load(f"{OUT}/{FILES[variant][1]}")
    for k in args.ks:
        lenses[f"routed_{variant}_k{k}"] = RoutedLens(experts, router, k)
    print("built routed lenses for", variant, flush=True)

_, items, excluded = jpp_cli.load_items_and_exclusions(
    model, eval_data_dir=jpp_cli.READOUT_EVALS_DATA_DIR, correctness_csv=CSV,
    hf_model_name=MODEL, layers=LAYERS, readout_max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN,
    items_spec="held-out:0",
)
ranks_df, pass_df = jpp_cli.evaluate(model, lenses, items, layers=LAYERS, excluded_token_ids=excluded, ks=[1, 10])
ranks_df.to_csv(args.out, index=False)
pass_path = args.out.replace(".csv", "_pass_at_k.csv")
pass_df.to_csv(pass_path, index=False)
print(pass_df[(pass_df["k"] == 10) & (pass_df["weighting"] == "item")].pivot(index="lens", columns="eval", values="pass_at_k").round(3).to_string())
print("wrote", args.out, pass_path)
