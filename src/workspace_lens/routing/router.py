from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch as t

from workspace_lens.routing.cluster import (
    fit_pca,
    kmeans,
    normalize_center_project,
    unit_normalize,
)
from workspace_lens.utils import get_position_mask_with_early_skips, token_ids_sha256


@dataclass
class LayerActivationRouter:
    """The frozen clustering for one layer: PCA projection plus centroids.

    With ``centroid_labels_C`` set, the "centroids" are labelled reference points
    (e.g. a subsample clustered by ADP) and an activation takes the label of its
    nearest reference point (1-nearest-neighbour) instead of the centroid's index."""

    pca_mean_N: t.Tensor
    pca_components_ND: t.Tensor
    centroids_ED: t.Tensor
    centroid_labels_C: t.Tensor | None = None

    def project(self, residual_BsN: t.Tensor) -> t.Tensor:
        """:func:`normalize_center_project` with this layer's parameters."""
        return normalize_center_project(
            residual_BsN,
            self.pca_mean_N.to(residual_BsN.device),
            self.pca_components_ND.to(residual_BsN.device),
        )

    def assign(self, residual_BsN: t.Tensor) -> t.Tensor:
        """Nearest-centroid cluster index for each activation: ``[...]`` (long)."""
        projected_BsD = self.project(residual_BsN)

        centroids_ED = self.centroids_ED.to(residual_BsN.device)

        flat_PD = projected_BsD.reshape(-1, projected_BsD.shape[-1])
        assignments_Int_P = t.cdist(flat_PD, centroids_ED).argmin(dim=1)
        if self.centroid_labels_C is not None:
            assignments_Int_P = self.centroid_labels_C.to(residual_BsN.device)[assignments_Int_P]
        return assignments_Int_P.reshape(projected_BsD.shape[:-1])


class ActivationRouterCollection:
    """Per-layer PCA + k-means clustering of residual activations.

    Fit once (:meth:`fit`) on pooled valid-position activations, then frozen:
    :class:`~workspace_lens.fitting.expert_fitting.ExpertJacobianTrainer` routes fitting
    positions through :meth:`assign`, one expert Jacobian per cluster. The router is
    used only while fitting: the J++ Lens combines the experts into one map per layer,
    so inference needs no router.
    """

    def __init__(
        self,
        num_clusters: int,
        projection_dim: int,
        *,
        seed: int = 0,
        kmeans_max_iterations: int = 100,
    ) -> None:
        self.num_clusters = num_clusters
        self.projection_dim = projection_dim
        self.seed = seed
        self.kmeans_max_iterations = kmeans_max_iterations
        self.layer_routers_L_dict: dict[int, LayerActivationRouter] = {}

    def fit(self, activations_L_dict_PN: dict[int, t.Tensor]) -> None:
        """Fit each layer's PCA and centroids from pooled activations
        ``{layer: [num_points, d_model]}``."""

        for layer, activations_PN in activations_L_dict_PN.items():

            normalized_PN = unit_normalize(activations_PN.float())

            pca_mean_N, pca_components_ND = fit_pca(normalized_PN, self.projection_dim)

            projected_PD = normalize_center_project(
                activations_PN, pca_mean_N, pca_components_ND
            )

            centroids_ED, _ = kmeans(
                projected_PD,
                self.num_clusters,
                seed=self.seed,
                max_iterations=self.kmeans_max_iterations,
            )

            self.layer_routers_L_dict[layer] = LayerActivationRouter(
                pca_mean_N=pca_mean_N,
                pca_components_ND=pca_components_ND,
                centroids_ED=centroids_ED,
            )

    def assign(self, residual_BsN: t.Tensor, layer: int) -> t.Tensor:
        """Cluster index for each activation at ``layer``: ``[...]`` (long)."""
        return self.layer_routers_L_dict[layer].assign(residual_BsN)

    ### SAVING AND LOADING (plain tensors only: weights_only=True safe)

    def save(self, path: str) -> None:
        t.save(
            {
                "num_clusters": self.num_clusters,
                "projection_dim": self.projection_dim,
                "seed": self.seed,
                "kmeans_max_iterations": self.kmeans_max_iterations,
                "pca_means": {
                    layer: layer_router.pca_mean_N.cpu()
                    for layer, layer_router in self.layer_routers_L_dict.items()
                },
                "pca_components": {
                    layer: layer_router.pca_components_ND.cpu()
                    for layer, layer_router in self.layer_routers_L_dict.items()
                },
                "centroids": {
                    layer: layer_router.centroids_ED.cpu()
                    for layer, layer_router in self.layer_routers_L_dict.items()
                },
                "centroid_labels": {
                    layer: layer_router.centroid_labels_C.cpu()
                    for layer, layer_router in self.layer_routers_L_dict.items()
                    if layer_router.centroid_labels_C is not None
                },
            },
            path,
        )

    @property
    def layers(self) -> list[int]:
        return sorted(self.layer_routers_L_dict)

    @classmethod
    def load(cls, path: str) -> ActivationRouterCollection:
        state = t.load(path, map_location="cpu", weights_only=True)
        router_collection = cls(
            num_clusters=state["num_clusters"],
            projection_dim=state["projection_dim"],
            seed=state["seed"],
            kmeans_max_iterations=state["kmeans_max_iterations"],
        )
        router_collection.layer_routers_L_dict = {
            layer: LayerActivationRouter(
                pca_mean_N=state["pca_means"][layer],
                pca_components_ND=state["pca_components"][layer],
                centroids_ED=state["centroids"][layer],
                centroid_labels_C=state.get("centroid_labels", {}).get(layer),
            )
            for layer in state["centroids"]
        }
        return router_collection


### A PARTITION THAT IGNORES THE ACTIVATIONS


class PositionPartitionRouter:
    """A fixed assignment of fit positions to ``num_clusters`` buckets, looked up from the
    prompt and the position instead of computed from the activation: a control for the
    clustering routers (:func:`random_position_partition` draws one with equal bucket sizes).

    A prompt is identified by the SHA-256 of its token ids
    (:func:`~workspace_lens.utils.token_ids_sha256`, as the fit's position records do).
    ``labels_L_dict_PS[layer][row, position]`` is the bucket of that position of the prompt
    ``prompt_hashes[row]`` (-1 where the position is not a fit position). The expert
    trainer calls :meth:`assign_prompt_positions`; :meth:`assign` from activations alone is
    an error."""

    kind = "position_partition"

    def __init__(
        self, num_clusters: int, prompt_hashes: Sequence[str], labels_L_dict_PS: dict[int, t.Tensor]
    ) -> None:
        self.num_clusters = num_clusters
        self.projection_dim = 0  # no projection: the config records 0
        self.prompt_hashes = list(prompt_hashes)
        self.labels_L_dict_PS = {layer: labels_PS.long() for layer, labels_PS in labels_L_dict_PS.items()}
        self._row_of_hash = {prompt_hash: row for row, prompt_hash in enumerate(self.prompt_hashes)}

    @property
    def layers(self) -> list[int]:
        return sorted(self.labels_L_dict_PS)

    def assign(self, residual_BsN: t.Tensor, layer: int) -> t.Tensor:
        raise TypeError("a PositionPartitionRouter assigns from the prompt and position: use assign_prompt_positions")

    def assign_prompt_positions(
        self, input_ids_Int_S: t.Tensor, positions_Int_P: t.Tensor, layer: int
    ) -> t.Tensor:
        """``[P]`` bucket of each of ``positions_Int_P`` in the prompt ``input_ids_Int_S``."""
        prompt_hash = token_ids_sha256(input_ids_Int_S)
        row = self._row_of_hash.get(prompt_hash)
        if row is None:
            raise KeyError(
                f"prompt {prompt_hash[:12]} is not in this partition (built for {len(self.prompt_hashes)} prompts)"
            )
        labels_S = self.labels_L_dict_PS[layer][row]
        labels_P = labels_S[positions_Int_P.cpu()]
        if bool((labels_P < 0).any()):
            raise ValueError(f"positions {positions_Int_P.tolist()} include ones this partition did not label")
        return labels_P.to(positions_Int_P.device)

    def save(self, path: str) -> None:
        t.save(
            {
                "kind": self.kind,
                "num_clusters": self.num_clusters,
                "prompt_hashes": self.prompt_hashes,
                "labels": {layer: labels_PS.cpu() for layer, labels_PS in self.labels_L_dict_PS.items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> PositionPartitionRouter:
        state = t.load(path, map_location="cpu", weights_only=True)
        return cls(state["num_clusters"], state["prompt_hashes"], state["labels"])


def random_position_partition(
    input_ids_list_S: Sequence[t.Tensor],
    *,
    num_clusters: int,
    layers: Sequence[int],
    skip_first_n_positions: int,
    seed: int,
) -> PositionPartitionRouter:
    """Every fit position of the prompts ``input_ids_list_S`` (tokenised as the fit
    tokenises them) assigned to one of ``num_clusters`` buckets at random, with the bucket
    sizes equal to within one: the positions are numbered in prompt order and bucket ``i mod
    K`` goes to a random permutation of the numbers, drawn independently per layer from
    ``seed``. Fit positions are those of
    :func:`~workspace_lens.utils.get_position_mask_with_early_skips`. Prompts too short to
    have one are left out (the fit skips them too)."""
    prompt_hashes: list[str] = []
    masks_list_S: list[t.Tensor] = []
    seen: set[str] = set()
    for input_ids_Int_S in input_ids_list_S:
        prompt_hash = token_ids_sha256(input_ids_Int_S)
        if prompt_hash in seen:
            continue  # the same prompt twice gets the same labels
        try:
            mask_Bool_S = get_position_mask_with_early_skips(len(input_ids_Int_S), skip_first_n_positions)
        except ValueError:
            continue
        seen.add(prompt_hash)
        prompt_hashes.append(prompt_hash)
        masks_list_S.append(mask_Bool_S)
    max_len = max(len(mask_Bool_S) for mask_Bool_S in masks_list_S)
    mask_Bool_PS = t.zeros(len(masks_list_S), max_len, dtype=t.bool)
    for row, mask_Bool_S in enumerate(masks_list_S):
        mask_Bool_PS[row, : len(mask_Bool_S)] = mask_Bool_S
    num_positions = int(mask_Bool_PS.sum())
    labels_L_dict_PS: dict[int, t.Tensor] = {}
    for layer in layers:
        generator = t.Generator().manual_seed(seed * 1_000_003 + layer)
        buckets_Int_N = (t.arange(num_positions) % num_clusters)[t.randperm(num_positions, generator=generator)]
        labels_PS = t.full((len(masks_list_S), max_len), -1, dtype=t.long)
        labels_PS[mask_Bool_PS] = buckets_Int_N
        labels_L_dict_PS[layer] = labels_PS
    return PositionPartitionRouter(num_clusters, prompt_hashes, labels_L_dict_PS)


def load_router(path: str) -> ActivationRouterCollection | PositionPartitionRouter:
    """A router file of either kind: a :class:`PositionPartitionRouter` (its file says so
    under ``"kind"``) or an :class:`ActivationRouterCollection`."""
    state = t.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") == PositionPartitionRouter.kind:
        return PositionPartitionRouter.load(path)
    return ActivationRouterCollection.load(path)
