"""PCA variance spectrum and TwoNN intrinsic dimension of the saved router activations.

Both run on the points the router sees before its projection: each activation scaled to
unit length (``unit_normalize``), in the full ``d_model`` space.

- PCA: the eigenvalues of the covariance (the router's ``fit_pca``, all components kept),
  so the cumulative variance explained by the top d directions can be read for every d.
- TwoNN (Facco et al. 2017): for each point, mu = r2 / r1, the ratio of its 2nd to 1st
  nearest-neighbour distance. Then -log(1 - F(mu)) = d * log(mu), where F is mu's
  empirical CDF; d is the slope of a line through the origin, fitted on the 90% smallest
  mu. Also run on random subsets (N/4, N/16) to see whether the estimate depends on scale.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch as t

from workspace_lens.routing.cluster import unit_normalize

logger = logging.getLogger(__name__)


@t.no_grad()
def two_nearest_distances(points_PN: t.Tensor, chunk_size: int = 4096) -> t.Tensor:
    """``[P, 2]``: distance from each point to its nearest and 2nd-nearest other point
    (float64, so an exact duplicate comes out at ~0 rather than at rounding noise)."""
    points_PN = points_PN.double()
    squared_norms_P = points_PN.pow(2).sum(dim=1)
    nearest_list_C2: list[t.Tensor] = []
    for start in range(0, points_PN.shape[0], chunk_size):
        chunk_CN = points_PN[start : start + chunk_size]
        squared_distances_CP = (
            squared_norms_P[start : start + chunk_size, None]
            + squared_norms_P[None, :]
            - 2 * chunk_CN @ points_PN.T
        )
        rows_C = t.arange(chunk_CN.shape[0], device=points_PN.device)
        squared_distances_CP[rows_C, start + rows_C] = float("inf")  # not its own neighbour
        nearest_C2 = squared_distances_CP.topk(2, dim=1, largest=False).values
        nearest_list_C2.append(nearest_C2.clamp(min=0).sqrt())
    return t.cat(nearest_list_C2)


def twonn_dimension(nearest_P2: t.Tensor, discard_fraction: float = 0.1) -> dict:
    """The TwoNN fit on ``[P, 2]`` nearest distances. Points with r1 ~ 0 (exact duplicates)
    are dropped first."""
    keep_Bool_P = nearest_P2[:, 0] > 1e-6
    mu_P = (nearest_P2[keep_Bool_P, 1] / nearest_P2[keep_Bool_P, 0]).double().sort().values
    num_points = mu_P.shape[0]
    empirical_cdf_P = t.arange(1, num_points + 1, dtype=t.float64, device=mu_P.device) / num_points
    num_fit = int(num_points * (1 - discard_fraction))
    log_mu_F = mu_P[:num_fit].log()
    neg_log_survival_F = -(1 - empirical_cdf_P[:num_fit]).log()
    dimension = float((log_mu_F * neg_log_survival_F).sum() / log_mu_F.pow(2).sum())
    return {
        "dimension": dimension,
        "num_points": num_points,
        "num_duplicates_dropped": int((~keep_Bool_P).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", default="artifacts/gemma4_e2b/activations")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28])
    parser.add_argument("--subsample-divisors", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-path", default="results/dimensionality/pca_and_twonn.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    device = "cuda" if t.cuda.is_available() else "cpu"
    results: dict = {"layers": {}}
    for layer in args.layers:
        activations_PN = t.load(
            Path(args.activations_dir) / f"layer{layer}.pt", weights_only=True
        )
        points_PN = unit_normalize(activations_PN.to(device).float())
        num_points = points_PN.shape[0]

        centered_PN = (points_PN - points_PN.mean(dim=0)).double()
        covariance_NN = centered_PN.T @ centered_PN / (num_points - 1)
        eigenvalues_N = t.linalg.eigvalsh(covariance_NN).flip(0).clamp(min=0)
        del centered_PN, covariance_NN

        twonn = {"all": twonn_dimension(two_nearest_distances(points_PN))}
        generator = t.Generator().manual_seed(args.seed)
        for divisor in args.subsample_divisors:
            subset_Int_S = t.randperm(num_points, generator=generator)[: num_points // divisor]
            twonn[f"1/{divisor}"] = twonn_dimension(
                two_nearest_distances(points_PN[subset_Int_S.to(device)])
            )

        results["layers"][str(layer)] = {
            "num_points": num_points,
            "eigenvalues": eigenvalues_N.cpu().tolist(),
            "twonn": twonn,
        }
        cumulative_N = eigenvalues_N.cumsum(0) / eigenvalues_N.sum()
        logger.info(
            "layer %d: variance in top 64 = %.3f; dims for 90/95/99%% = %s; TwoNN %s",
            layer,
            float(cumulative_N[63]),
            [int((cumulative_N < q).sum()) + 1 for q in (0.9, 0.95, 0.99)],
            {name: round(fit["dimension"], 1) for name, fit in twonn.items()},
        )
        logger.info("  duplicates dropped: %s", {n: f["num_duplicates_dropped"] for n, f in twonn.items()})

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results))
    logger.info("saved %s", results_path)


if __name__ == "__main__":
    main()
