"""Save the activations fit-router clusters, then sweep k-means K on their 64-dim projection.

1. Re-record the fit-router pool exactly: the same WikiText prompts, layers, truncation and
   valid positions (``collect_valid_position_activations``' loop, kept here so each row's
   prompt index, position and token id can be saved alongside it).
2. Save per layer the raw residuals ``[P, d_model]`` in the model's dtype (bf16: lossless),
   the router's 64-dim projection ``[P, D]`` (fp32), and the row index.
3. Check the recording against the saved router: refit PCA + k-means (K, seed and device
   as fit-router used) and compare with ``router.pt``.
4. k-means (the router's own ``kmeans``) on the projection for each K and seed; save the
   loss (inertia: sum of squared distances from each point to its nearest centre).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch as t

from jlens.examples import load_wikitext_prompts
from workspace_lens import get_hf_model
from workspace_lens.routing.cluster import kmeans
from workspace_lens.routing.router import ActivationRouterCollection
from workspace_lens.utils import get_position_mask_with_early_skips, record_activations

logger = logging.getLogger(__name__)


def record_pool(model, prompts, layers, *, skip_first_n_positions, max_seq_len):
    """``({layer: [P, N]}, {"prompt_idx", "position", "token_id": [P]})`` over the valid
    positions of ``prompts``, in the order ``collect_valid_position_activations`` uses."""
    activations_L_dict_list_PN: dict[int, list[t.Tensor]] = {layer: [] for layer in layers}
    index_list: dict[str, list[t.Tensor]] = {"prompt_idx": [], "position": [], "token_id": []}
    for prompt_idx, prompt in enumerate(prompts):
        input_ids_Int_1S, activations_L_dict_SN = record_activations(
            model, prompt, max_seq_len, list(layers)
        )
        try:
            mask_Bool_S = get_position_mask_with_early_skips(
                input_ids_Int_1S.shape[1], skip_first_n_positions
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            continue
        source_positions_Int_P = t.where(mask_Bool_S)[0]
        for layer in layers:
            activations_L_dict_list_PN[layer].append(
                activations_L_dict_SN[layer][source_positions_Int_P].cpu()
            )
        index_list["prompt_idx"].append(t.full_like(source_positions_Int_P, prompt_idx))
        index_list["position"].append(source_positions_Int_P)
        index_list["token_id"].append(input_ids_Int_1S[0].cpu()[source_positions_Int_P])
    activations_L_dict_PN = {
        layer: t.cat(activations_L_dict_list_PN[layer]) for layer in layers
    }
    index = {name: t.cat(parts) for name, parts in index_list.items()}
    return activations_L_dict_PN, index


def inertia(points_PD: t.Tensor, centroids_ED: t.Tensor, assignments_Int_P: t.Tensor) -> float:
    """Sum over points of the squared distance to the assigned centre."""
    return float((points_PD - centroids_ED[assignments_Int_P]).pow(2).sum())


def subspace_overlap(components_a_ND: t.Tensor, components_b_ND: t.Tensor) -> float:
    """Mean squared cosine of the principal angles between two D-dim subspaces
    (1 = same subspace, D/N for random ones)."""
    singular_values_D = t.linalg.svdvals(components_a_ND.T @ components_b_ND)
    return float(singular_values_D.pow(2).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--router-path", default="artifacts/gemma4_e2b/router.pt")
    parser.add_argument("--router-stamp", default="artifacts/gemma4_e2b/router_stamp.json")
    parser.add_argument("--activations-dir", default="artifacts/gemma4_e2b/activations")
    parser.add_argument("--results-path", default="results/kmeans_k_sweep/sweep.json")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # The settings fit-router ran with, read from its stamp so the pool is the same one.
    router_args = json.loads(Path(args.router_stamp).read_text())["args"]
    layers = router_args["layers"]
    router = ActivationRouterCollection.load(args.router_path)
    assert sorted(router.layer_routers_L_dict) == sorted(layers)

    model = get_hf_model(args.hf_model_name, attn_implementation=router_args["attn_implementation"])
    # load_fit_prompts(n) in jpp_cli: the first n of a pool of at least 1000.
    num_prompts = router_args["num_prompts"]
    prompts = load_wikitext_prompts(max(1000, num_prompts))[:num_prompts]

    start = time.time()
    activations_L_dict_PN, index = record_pool(
        model,
        prompts,
        layers,
        skip_first_n_positions=router_args["skip_first_n_positions"],
        max_seq_len=router_args["max_seq_len"],
    )
    num_points, d_model = activations_L_dict_PN[layers[0]].shape
    logger.info(
        "recorded %d positions x %d dims per layer (%s) in %.0fs",
        num_points, d_model, activations_L_dict_PN[layers[0]].dtype, time.time() - start,
    )
    del model
    t.cuda.empty_cache()

    activations_dir = Path(args.activations_dir)
    activations_dir.mkdir(parents=True, exist_ok=True)
    t.save(index, activations_dir / "index.pt")
    (activations_dir / "prompts.json").write_text(json.dumps(prompts))

    device = "cuda" if t.cuda.is_available() else "cpu"
    results: dict = {
        "hf_model_name": args.hf_model_name,
        "router_path": args.router_path,
        "num_points": num_points,
        "d_model": d_model,
        "projection_dim": router.projection_dim,
        "ks": args.ks,
        "seeds": args.seeds,
        "layers": {},
    }
    for layer in layers:
        activations_PN = activations_L_dict_PN[layer]
        t.save(activations_PN, activations_dir / f"layer{layer}.pt")
        layer_router = router.layer_routers_L_dict[layer]
        projected_PD = layer_router.project(activations_PN)  # fp32 [P, D]
        t.save(projected_PD, activations_dir / f"proj{router.projection_dim}_layer{layer}.pt")

        # Recording check: fit-router's own fit (CPU, fp32) on these points vs router.pt.
        refit = ActivationRouterCollection(
            router.num_clusters, router.projection_dim, seed=router.seed,
            kmeans_max_iterations=router.kmeans_max_iterations,
        )
        refit.fit({layer: activations_PN.float()})
        refit_router = refit.layer_routers_L_dict[layer]
        check = {
            "pca_subspace_overlap": subspace_overlap(
                layer_router.pca_components_ND, refit_router.pca_components_ND
            ),
            "pca_mean_max_abs_diff": float(
                (layer_router.pca_mean_N - refit_router.pca_mean_N).abs().max()
            ),
            "centroid_max_abs_diff": float(
                (layer_router.centroids_ED - refit_router.centroids_ED).abs().max()
            ),
        }
        saved_assignments_Int_P = layer_router.assign(activations_PN)
        check["same_cluster_fraction"] = float(
            (saved_assignments_Int_P == refit_router.assign(activations_PN)).float().mean()
        )
        saved_router_inertia = inertia(
            projected_PD, layer_router.centroids_ED, saved_assignments_Int_P
        )

        points_PD = projected_PD.to(device)
        k1_inertia = float((points_PD - points_PD.mean(dim=0)).pow(2).sum())
        inertia_by_k: dict[str, list[float]] = {}
        for k in args.ks:
            inertia_by_k[str(k)] = []
            for seed in args.seeds:
                centroids_ED, assignments_Int_P = kmeans(points_PD, k, seed=seed)
                inertia_by_k[str(k)].append(inertia(points_PD, centroids_ED, assignments_Int_P))
        results["layers"][str(layer)] = {
            "inertia": inertia_by_k,
            "k1_inertia": k1_inertia,
            "saved_router_inertia": saved_router_inertia,
            "check": check,
        }
        logger.info(
            "layer %d: check %s | saved-router K=%d inertia %.1f | %s",
            layer, json.dumps(check), router.num_clusters, saved_router_inertia,
            {k: round(min(v), 1) for k, v in inertia_by_k.items()},
        )

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("saved %s and %s", activations_dir, results_path)


if __name__ == "__main__":
    main()
