"""Fitting a lens by gradient descent on intervention pairs.

The lens reads a layer-``l`` residual like the J-lens, ``unembed(A_l @ h)``
(:class:`~workspace_lens.lenses.jacobian_lens.JacobianLens`), but ``A_l`` is learned
from edits instead of averaged from gradients:

1. Edit the output of block ``l`` at one kept position ``s`` of a prompt,
   ``h -> h + delta`` (:class:`EditFamily` says how ``delta`` is drawn).
2. Run the rest of the model and record the change in the output of the target block.
   The target ``y`` is that change summed over the kept positions ``t >= s`` (the
   structure of the J-lens estimator in :mod:`jacobian_fitting`: its row at ``s`` is
   ``sum_{kept t >= s} dh_target[t] / dh_l[s]``), or the change at ``t = s`` only.
3. Fit ``A_l`` to the pairs ``(delta, y)`` by Adam on a squared error
   (:func:`fit_transport`), from the identity or from a given lens.

With tiny Gaussian edits the least-squares ``A_l`` is the mean Jacobian, i.e. the
J-lens: the sanity check. With real-size edits (swaps between positions) it is not.

:func:`fit_output_kl` is the Tuned-Lens-style baseline on the same clean passes:
``A_l`` fitted so that ``unembed(A_l h)`` matches the model's own next-token
distribution (KL), with no edits at all.

Shapes: ``R`` rows of the clean pool (one per kept position of every prompt), ``E``
edited rows of one prompt, ``P`` pairs, ``N`` / ``F`` source / target width.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch as t
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel
from workspace_lens.config import LensConfig
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.residual_streams import has_residual_streams, residual_mean_over_streams
from workspace_lens.utils import get_position_mask_with_early_skips, record_block_outputs

logger = logging.getLogger(__name__)

EditKind = Literal["swap", "gauss", "zero", "mean"]
TargetKind = Literal["sum_future", "same_position"]
LossKind = Literal["residual", "logit"]


### EDIT FAMILIES


@dataclass(frozen=True)
class EditFamily:
    """How the edit ``delta`` at ``(layer, s)`` is drawn.

    - ``swap``: ``delta = alpha (h' - h)``, ``h'`` the layer's residual at a random
      kept position of another prompt (``alpha = 1`` is a full swap).
    - ``gauss``: ``delta = size * |h| / sqrt(d) * eps``, ``eps`` standard normal, so
      ``|delta| ~ size * |h|``. Gauss pairs are stored divided by that per-pair scale,
      so every pair weighs the same in the fit and its optimum is the unweighted mean
      Jacobian (the J-lens) rather than one weighted by ``|h|^2``.
    - ``zero``: ``delta = -h``. ``mean``: ``delta = mean_l - h`` (the mean over the
      clean pool's kept positions).
    """

    kind: EditKind
    alpha: float = 1.0
    size: float = 0.05

    @property
    def label(self) -> str:
        """Directory name of the family's pairs: ``swap_a1``, ``gauss_s0.05``, ``zero``."""
        if self.kind == "swap":
            return f"swap_a{self.alpha:g}"
        if self.kind == "gauss":
            return f"gauss_s{self.size:g}"
        return self.kind

    @classmethod
    def parse(cls, text: str) -> EditFamily:
        """``"swap:0.5"``, ``"swap"`` (alpha 1), ``"gauss:0.05"``, ``"gauss"`` (size
        0.05), ``"zero"``, ``"mean"``."""
        kind, _, value = text.partition(":")
        if kind == "swap":
            return cls("swap", alpha=float(value) if value else 1.0)
        if kind == "gauss":
            return cls("gauss", size=float(value) if value else 0.05)
        if kind in ("zero", "mean") and not value:
            return cls(kind)  # type: ignore[arg-type]
        raise ValueError(f"unknown edit family {text!r}")


### THE CLEAN POOL


@dataclass
class CleanPool:
    """Clean-run residuals at every kept position of every prompt: the donors of the
    swap family, the per-layer means of the ``mean`` family, and the training data
    of :func:`fit_output_kl`. Row ``r`` is position ``position_Int_R[r]`` of prompt
    ``prompt_idx_Int_R[r]``; rows of one prompt are contiguous. bf16 on the CPU."""

    prompt_idx_Int_R: t.Tensor
    position_Int_R: t.Tensor
    sources_L_dict_RN: dict[int, t.Tensor]
    # The lens's residual at the target block (the final block for relative offset -1).
    target_RF: t.Tensor
    num_prompts: int

    @property
    def layers(self) -> list[int]:
        return sorted(self.sources_L_dict_RN)

    def save(self, path: str | Path) -> None:
        t.save(
            {
                "prompt_idx": self.prompt_idx_Int_R,
                "position": self.position_Int_R,
                "sources": self.sources_L_dict_RN,
                "target": self.target_RF,
                "num_prompts": self.num_prompts,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> CleanPool:
        state = t.load(path, map_location="cpu", weights_only=True)
        return cls(
            prompt_idx_Int_R=state["prompt_idx"],
            position_Int_R=state["position"],
            sources_L_dict_RN=state["sources"],
            target_RF=state["target"],
            num_prompts=int(state["num_prompts"]),
        )


def record_clean_pool(
    model: LensModel,
    prompts: Sequence[str],
    *,
    layers: Sequence[int],
    target_layer: int,
    max_seq_len: int,
    skip_first_n_positions: int,
) -> CleanPool:
    """One clean forward per prompt; the lens's residual at ``layers`` and at
    ``target_layer`` at every kept position
    (:func:`~workspace_lens.utils.get_position_mask_with_early_skips`). Prompts too
    short to keep a position are skipped with a warning (and so get no pairs)."""
    prompt_idx_list: list[t.Tensor] = []
    position_list: list[t.Tensor] = []
    sources_L_dict_list: dict[int, list[t.Tensor]] = {layer: [] for layer in layers}
    target_list: list[t.Tensor] = []
    for prompt_idx, prompt in enumerate(prompts):
        input_ids_Int_1S, block_outputs_L_dict_1SrN = record_block_outputs(
            model, prompt, max_seq_len, sorted({*layers, target_layer})
        )
        try:
            kept_Int_P = t.where(
                get_position_mask_with_early_skips(input_ids_Int_1S.shape[1], skip_first_n_positions)
            )[0]
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            continue
        prompt_idx_list.append(t.full((len(kept_Int_P),), prompt_idx, dtype=t.long))
        position_list.append(kept_Int_P)
        for layer in layers:
            residual_SN = residual_mean_over_streams(block_outputs_L_dict_1SrN[layer])[0]
            sources_L_dict_list[layer].append(residual_SN[kept_Int_P.to(residual_SN.device)].bfloat16().cpu())
        target_SF = residual_mean_over_streams(block_outputs_L_dict_1SrN[target_layer])[0]
        target_list.append(target_SF[kept_Int_P.to(target_SF.device)].bfloat16().cpu())
    if not prompt_idx_list:
        raise ValueError("no prompts were long enough to keep a position")
    return CleanPool(
        prompt_idx_Int_R=t.cat(prompt_idx_list),
        position_Int_R=t.cat(position_list),
        sources_L_dict_RN={layer: t.cat(rows) for layer, rows in sources_L_dict_list.items()},
        target_RF=t.cat(target_list),
        num_prompts=len(prompts),
    )


### PAIRS


@dataclass
class LayerPairs:
    """One family's pairs at one source layer. ``delta_PN`` is the edit actually
    applied (the bf16 edited residual minus the clean one, so the rounding of the
    edited residual is part of it); ``y_sum_PF`` the target change summed over the
    kept positions ``t >= s``; ``y_same_PF`` the change at ``t = s``. Gauss pairs are
    all three divided by the pair's noise scale (:class:`EditFamily`). bf16 on the
    CPU, plus the prompt index and edited position of each pair."""

    delta_PN: t.Tensor
    y_sum_PF: t.Tensor
    y_same_PF: t.Tensor
    prompt_idx_Int_P: t.Tensor
    position_Int_P: t.Tensor

    def target(self, kind: TargetKind) -> t.Tensor:
        return self.y_sum_PF if kind == "sum_future" else self.y_same_PF

    def save(self, path: str | Path) -> None:
        t.save(
            {
                "delta": self.delta_PN,
                "y_sum": self.y_sum_PF,
                "y_same": self.y_same_PF,
                "prompt_idx": self.prompt_idx_Int_P,
                "position": self.position_Int_P,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LayerPairs:
        state = t.load(path, map_location="cpu", weights_only=True)
        return cls(
            delta_PN=state["delta"],
            y_sum_PF=state["y_sum"],
            y_same_PF=state["y_same"],
            prompt_idx_Int_P=state["prompt_idx"],
            position_Int_P=state["position"],
        )

    @classmethod
    def concatenate(cls, parts: Sequence[LayerPairs]) -> LayerPairs:
        return cls(
            *(t.cat([getattr(part, name) for part in parts]) for name in cls.__dataclass_fields__)
        )


def pairs_path(pairs_dir: str | Path, layer: int) -> Path:
    return Path(pairs_dir) / f"pairs_layer{layer}.pt"


@dataclass
class _PromptEdits:
    """Every edited row of one prompt, flattened over (family, layer, pair):
    which block and position each row edits, and how (``edited = h + alpha (anchor -
    h) + noise_size |h| / sqrt(d) noise``)."""

    family_idx_Int_E: t.Tensor
    layer_Int_E: t.Tensor
    position_Int_E: t.Tensor
    anchor_EN: t.Tensor
    alpha_E: t.Tensor
    noise_EN: t.Tensor
    noise_size_E: t.Tensor

    def chunk(self, rows: slice) -> _PromptEdits:
        return _PromptEdits(*(getattr(self, name)[rows] for name in self.__dataclass_fields__))

    @property
    def num_rows(self) -> int:
        return int(self.layer_Int_E.numel())


class RowEditHooks:
    """Writer forward hooks that edit one position of one batch row per edit: row
    ``row_offset + e`` of block ``layer_Int_E[e]``'s output at position
    ``position_Int_E[e]`` becomes ``h + alpha (anchor - h) + noise_size |h| / sqrt(d)
    noise``, computed in fp32 and cast back to the residual's dtype. Rows before
    ``row_offset`` (the clean reference) are untouched. After the forward,
    ``effective_delta_EN`` holds the edits actually applied (edited minus clean, after
    the cast) and ``noise_scale_E`` each row's ``noise_size |h| / sqrt(d)``.

    The per-row counterpart of
    :class:`~workspace_lens.interventions.hooks.InterventionHooks`, which applies
    one edit to every row."""

    def __init__(self, model: LensModel, edits: _PromptEdits, *, row_offset: int = 1) -> None:
        self.model = model
        self.edits = edits
        self.row_offset = row_offset
        device = edits.anchor_EN.device
        self.effective_delta_EN = t.zeros_like(edits.anchor_EN)
        self.noise_scale_E = t.zeros(edits.num_rows, device=device)
        self._handles: list[t.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int) -> Callable[..., t.Tensor | tuple]:
        rows_Int_e = t.where(self.edits.layer_Int_E == layer)[0]

        def hook(module: nn.Module, inputs: tuple, output: t.Tensor | tuple) -> t.Tensor | tuple:
            block_output_BSN = output if t.is_tensor(output) else output[0]
            if has_residual_streams(block_output_BSN):
                raise NotImplementedError("row edits do not lift to a multi-stream residual")
            rows_on_device_e = rows_Int_e.to(block_output_BSN.device)
            batch_rows_e = rows_on_device_e + self.row_offset
            positions_e = self.edits.position_Int_E[rows_on_device_e]
            clean_eN = block_output_BSN[batch_rows_e, positions_e].float()
            noise_scale_e = (
                self.edits.noise_size_E[rows_on_device_e]
                * clean_eN.norm(dim=-1)
                / math.sqrt(clean_eN.shape[-1])
            )
            edited_eN = (
                clean_eN
                + self.edits.alpha_E[rows_on_device_e, None]
                * (self.edits.anchor_EN[rows_on_device_e] - clean_eN)
                + noise_scale_e[:, None] * self.edits.noise_EN[rows_on_device_e]
            ).to(block_output_BSN.dtype)
            self.effective_delta_EN[rows_on_device_e] = edited_eN.float() - clean_eN
            self.noise_scale_E[rows_on_device_e] = noise_scale_e
            edited_BSN = block_output_BSN.clone()
            edited_BSN[batch_rows_e, positions_e] = edited_eN
            if t.is_tensor(output):
                return edited_BSN
            return (edited_BSN, *output[1:])

        return hook

    def __enter__(self) -> RowEditHooks:
        try:
            for layer in t.unique(self.edits.layer_Int_E).tolist():
                self._handles.append(
                    self.model.layers[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _prompt_generator(seed: int, prompt_idx: int) -> t.Generator:
    """The random draws of one prompt depend only on ``(seed, prompt_idx)``, so pairs
    collected in separate runs share their positions, donors and noise (the swap
    families with different alphas are paired)."""
    return t.Generator().manual_seed(seed * 1_000_003 + prompt_idx)


def _draw_prompt_edits(
    pool: CleanPool,
    prompt_idx: int,
    kept_Int_P: t.Tensor,
    *,
    families: Sequence[EditFamily],
    layers: Sequence[int],
    pairs_per_prompt: int,
    layer_means_L_dict_N: dict[int, t.Tensor],
    seed: int,
    device: t.device | str,
) -> _PromptEdits:
    """The edited rows of one prompt. Per layer, ``pairs_per_prompt`` positions,
    donor rows (a uniform row of the pool from another prompt) and noise vectors are
    drawn once and shared by every family."""
    generator = _prompt_generator(seed, prompt_idx)
    num_layers, num_pairs = len(layers), pairs_per_prompt
    d_model = pool.sources_L_dict_RN[layers[0]].shape[-1]

    positions_Int_LB = kept_Int_P[t.randint(len(kept_Int_P), (num_layers, num_pairs), generator=generator)]
    # A uniform row among the rows of the other prompts (this prompt's rows are contiguous).
    own_rows_Int = t.where(pool.prompt_idx_Int_R == prompt_idx)[0]
    num_other_rows = len(pool.prompt_idx_Int_R) - len(own_rows_Int)
    if num_other_rows == 0:
        raise ValueError("the swap family needs at least two prompts in the pool")
    donor_rows_Int_LB = t.randint(num_other_rows, (num_layers, num_pairs), generator=generator)
    if len(own_rows_Int):
        donor_rows_Int_LB = donor_rows_Int_LB + (donor_rows_Int_LB >= own_rows_Int[0]) * len(own_rows_Int)
    noise_LBN = t.randn(num_layers, num_pairs, d_model, generator=generator)

    parts: dict[str, list[t.Tensor]] = {name: [] for name in _PromptEdits.__dataclass_fields__}
    for family_idx, family in enumerate(families):
        for layer_idx, layer in enumerate(layers):
            if family.kind == "swap":
                anchor_BN = pool.sources_L_dict_RN[layer][donor_rows_Int_LB[layer_idx]].float()
            elif family.kind == "mean":
                anchor_BN = layer_means_L_dict_N[layer].expand(num_pairs, -1)
            else:
                anchor_BN = t.zeros(num_pairs, d_model)
            alpha = family.alpha if family.kind == "swap" else (0.0 if family.kind == "gauss" else 1.0)
            noise_size = family.size if family.kind == "gauss" else 0.0
            parts["family_idx_Int_E"].append(t.full((num_pairs,), family_idx, dtype=t.long))
            parts["layer_Int_E"].append(t.full((num_pairs,), layer, dtype=t.long))
            parts["position_Int_E"].append(positions_Int_LB[layer_idx])
            parts["anchor_EN"].append(anchor_BN)
            parts["alpha_E"].append(t.full((num_pairs,), alpha))
            parts["noise_EN"].append(noise_LBN[layer_idx])
            parts["noise_size_E"].append(t.full((num_pairs,), noise_size))
    return _PromptEdits(**{name: t.cat(tensors).to(device) for name, tensors in parts.items()})


@dataclass
class PairCollectionStats:
    """Per-prompt diagnostics of :func:`collect_pairs`. ``max_leak`` is the largest
    change seen at a position *before* the edited one (causally impossible, so it is
    the run-to-run noise between batch rows: 0 when the kernels are row-invariant);
    ``max_abs_y`` the largest target entry (for the storage dtype)."""

    seconds_per_prompt: list[float] = field(default_factory=list)
    max_leak: float = 0.0
    max_abs_y: float = 0.0


def collect_pairs(
    model: LensModel,
    prompts: Sequence[str],
    pool: CleanPool,
    *,
    families: Sequence[EditFamily],
    layers: Sequence[int],
    target_layer: int,
    pairs_per_prompt: int,
    rows_per_pass: int,
    max_seq_len: int,
    skip_first_n_positions: int,
    seed: int = 0,
    log_every: int = 16,
) -> tuple[dict[str, dict[int, LayerPairs]], PairCollectionStats]:
    """Edit pairs for every family, ``pairs_per_prompt`` per prompt and layer:
    ``{family.label: {layer: LayerPairs}}``.

    Per prompt, every (family, layer, pair) edit is one batch row; rows go through the
    model ``rows_per_pass - 1`` at a time behind one clean row, so each edited row is
    compared against a clean run in the same batch. Each row edits one position of one
    block's output (:class:`RowEditHooks`); the target block's output is recorded and
    differenced against the clean row. Prompts the pool skipped get no pairs."""
    if rows_per_pass < 2:
        raise ValueError("rows_per_pass must leave room for the clean row and one edit")
    device = getattr(model, "input_device", t.device("cpu"))
    layer_means_L_dict_N = {
        layer: pool.sources_L_dict_RN[layer].float().mean(dim=0) for layer in layers
    }
    pooled_prompts = sorted(set(pool.prompt_idx_Int_R.tolist()))
    collected: dict[str, dict[int, list[LayerPairs]]] = {
        family.label: {layer: [] for layer in layers} for family in families
    }
    stats = PairCollectionStats()

    for count, prompt_idx in enumerate(pooled_prompts):
        prompt_start_time = time.perf_counter()
        input_ids_Int_1S = model.encode(prompts[prompt_idx], max_length=max_seq_len)
        seq_len = input_ids_Int_1S.shape[1]
        kept_mask_Bool_S = get_position_mask_with_early_skips(seq_len, skip_first_n_positions)
        kept_Int_P = t.where(kept_mask_Bool_S)[0]
        edits = _draw_prompt_edits(
            pool,
            prompt_idx,
            kept_Int_P,
            families=families,
            layers=layers,
            pairs_per_prompt=pairs_per_prompt,
            layer_means_L_dict_N=layer_means_L_dict_N,
            seed=seed,
            device=device,
        )
        kept_mask_on_device_Bool_S = kept_mask_Bool_S.to(device)
        sequence_Int_S = t.arange(seq_len, device=device)

        for start in range(0, edits.num_rows, rows_per_pass - 1):
            chunk = edits.chunk(slice(start, start + rows_per_pass - 1))
            hooks = RowEditHooks(model, chunk, row_offset=1)
            with (
                t.no_grad(),
                hooks,
                ActivationRecorder(model.layers, at=[target_layer]) as recorder,
            ):
                model.forward(input_ids_Int_1S.expand(chunk.num_rows + 1, -1))
                target_BSF = residual_mean_over_streams(recorder.activations[target_layer])
                change_ESF = target_BSF[1:].float() - target_BSF[:1].float()

            future_Bool_ES = (sequence_Int_S[None, :] >= chunk.position_Int_E[:, None]) & kept_mask_on_device_Bool_S
            y_sum_EF = t.einsum("esf,es->ef", change_ESF, future_Bool_ES.float())
            y_same_EF = change_ESF[t.arange(chunk.num_rows, device=device), chunk.position_Int_E]
            before_Bool_ES = sequence_Int_S[None, :] < chunk.position_Int_E[:, None]
            if before_Bool_ES.any():
                stats.max_leak = max(
                    stats.max_leak, float(change_ESF.abs().amax(dim=-1)[before_Bool_ES].max())
                )
            stats.max_abs_y = max(stats.max_abs_y, float(y_sum_EF.abs().max()))

            # Gauss rows are stored per unit of their noise scale; every other row as is.
            scale_E = t.where(chunk.noise_size_E > 0, hooks.noise_scale_E, t.ones_like(hooks.noise_scale_E))
            for family_idx, family in enumerate(families):
                for layer in layers:
                    rows_Int_e = t.where((chunk.family_idx_Int_E == family_idx) & (chunk.layer_Int_E == layer))[0]
                    if not len(rows_Int_e):
                        continue
                    scale_e1 = scale_E[rows_Int_e, None]
                    collected[family.label][layer].append(
                        LayerPairs(
                            delta_PN=(hooks.effective_delta_EN[rows_Int_e] / scale_e1).bfloat16().cpu(),
                            y_sum_PF=(y_sum_EF[rows_Int_e] / scale_e1).bfloat16().cpu(),
                            y_same_PF=(y_same_EF[rows_Int_e] / scale_e1).bfloat16().cpu(),
                            prompt_idx_Int_P=t.full((len(rows_Int_e),), prompt_idx, dtype=t.long),
                            position_Int_P=chunk.position_Int_E[rows_Int_e].cpu(),
                        )
                    )

        stats.seconds_per_prompt.append(time.perf_counter() - prompt_start_time)
        if (count + 1) % log_every == 0 or count + 1 == len(pooled_prompts):
            logger.info(
                "  prompt %d/%d  %.2fs/prompt  max_leak=%.3g  max|y|=%.3g",
                count + 1,
                len(pooled_prompts),
                sum(stats.seconds_per_prompt[-log_every:]) / len(stats.seconds_per_prompt[-log_every:]),
                stats.max_leak,
                stats.max_abs_y,
            )

    return {
        label: {layer: LayerPairs.concatenate(parts) for layer, parts in per_layer.items()}
        for label, per_layer in collected.items()
    }, stats


def projected_jacobian_errors(
    model: LensModel,
    prompt: str,
    pairs: LayerPairs,
    *,
    layer: int,
    target_layer: int,
    max_seq_len: int,
    skip_first_n_positions: int,
    num_projections: int = 8,
    seed: int = 0,
) -> t.Tensor:
    """How far each pair's summed target ``y`` is from the J-lens estimator's
    first-order prediction ``G_s delta``, ``G_s = sum_{kept t >= s} dh_target[t] /
    dh_layer[s]``: one relative error per pair, measured along ``num_projections``
    random directions ``u`` (``u . y`` against ``(G_s^T u) . delta``). ``G_s^T u`` is
    the J-lens estimator's backward pass with the cotangent ``u`` at every kept
    position, read at ``s``. Near zero for small Gaussian edits when the edit
    position, the hook and the summed positions match the J-lens; for swaps it
    measures how far a real-size edit is from linear. ``pairs`` must all come from
    ``prompt`` (scaling a pair leaves its error unchanged, so stored gauss pairs work
    as is). Single-stream models only."""
    input_ids_Int_1S = model.encode(prompt, max_length=max_seq_len)
    kept_Int_P = t.where(
        get_position_mask_with_early_skips(input_ids_Int_1S.shape[1], skip_first_n_positions)
    )[0]
    with (
        ActivationRecorder(model.layers, at=[layer, target_layer], start_graph_at=layer) as recorder,
        t.enable_grad(),
    ):
        model.forward(input_ids_Int_1S.expand(num_projections, -1))
        target_BSF = recorder.activations[target_layer]
        source_BSN = recorder.activations[layer]
    # Rounded to the residual's dtype first, so the cotangent and the projection of y agree.
    directions_KF = (
        t.randn(num_projections, target_BSF.shape[-1], generator=t.Generator().manual_seed(seed))
        .to(device=target_BSF.device, dtype=target_BSF.dtype)
        .float()
    )
    cotangent_BSF = t.zeros_like(target_BSF)
    cotangent_BSF[:, kept_Int_P.to(target_BSF.device), :] = directions_KF[:, None, :].to(target_BSF.dtype)
    (gradient_BSN,) = t.autograd.grad(target_BSF, source_BSN, cotangent_BSF)
    positions_Int_P = pairs.position_Int_P.to(gradient_BSN.device)
    predicted_PK = t.einsum(
        "kpn,pn->pk", gradient_BSN[:, positions_Int_P, :].float(), pairs.delta_PN.to(gradient_BSN.device).float()
    )
    measured_PK = pairs.y_sum_PF.to(gradient_BSN.device).float() @ directions_KF.T
    return ((measured_PK - predicted_PK).norm(dim=-1) / predicted_PK.norm(dim=-1)).cpu()


### THE LOGIT-SPACE METRIC


@t.no_grad()
def unembed_gram(model: LensModel, *, vocab_chunk: int = 32768) -> t.Tensor:
    """``M = diag(g) W_U^T W_U diag(g)`` (``[F, F]`` fp32, CPU), so ``x^T M x`` is the
    squared norm of the logits ``W_U diag(g) x`` (no soft-cap, no ``1 / rms``). ``g``
    is read by passing a vector of ones through the final norm, which assumes an
    RMSNorm-type norm (true for Gemma, Qwen, Llama). Needs the HF adapter's
    ``_final_norm`` and ``_lm_head``."""
    final_norm = model._final_norm  # type: ignore[attr-defined]
    lm_head_weight_VF = model._lm_head.weight  # type: ignore[attr-defined]
    ones_F = t.ones(model.d_model, device=lm_head_weight_VF.device, dtype=t.float32)
    gain_F = final_norm(ones_F[None, :])[0].float()
    gram_FF = t.zeros(model.d_model, model.d_model, device=lm_head_weight_VF.device)
    for start in range(0, lm_head_weight_VF.shape[0], vocab_chunk):
        weight_chunk_VF = lm_head_weight_VF[start : start + vocab_chunk].float() * gain_F
        gram_FF += weight_chunk_VF.T @ weight_chunk_VF
    return gram_FF.cpu()


### FITTING A TRANSPORT MAP


def heldout_prompt_mask(num_prompts: int, fraction: float, *, seed: int = 0) -> t.Tensor:
    """``Bool[num_prompts]``: a seeded random ``fraction`` of the prompts (at least one)
    held out for early stopping."""
    num_heldout = max(1, round(fraction * num_prompts))
    order = t.randperm(num_prompts, generator=t.Generator().manual_seed(seed))
    mask_Bool = t.zeros(num_prompts, dtype=t.bool)
    mask_Bool[order[:num_heldout]] = True
    return mask_Bool


@dataclass
class TransportFit:
    """One layer's fitted map and how it got there. Errors are relative:
    ``sum ||A delta - y||^2 / sum ||y||^2`` over the held-out pairs (in the
    ``M``-norm for the logit loss), at the init and at the kept (best) step.
    ``lstsq_heldout_relative_error`` is the same error for the closed-form least-squares
    map on the training pairs (``nan`` when not computed): a check of how close Adam got
    to the optimum. Both losses share that optimum, since an unconstrained linear map
    minimises every positive-definite output metric at once."""

    matrix_FN: t.Tensor
    best_step: int
    init_heldout_relative_error: float
    best_heldout_relative_error: float
    history: list[dict[str, float]]
    lstsq_heldout_relative_error: float = float("nan")


def _squared_error_P(residual_PF: t.Tensor, gram_FF: t.Tensor | None) -> t.Tensor:
    if gram_FF is None:
        return residual_PF.pow(2).sum(dim=-1)
    return ((residual_PF @ gram_FF) * residual_PF).sum(dim=-1)


def fit_transport(
    delta_PN: t.Tensor,
    y_PF: t.Tensor,
    heldout_Bool_P: t.Tensor,
    *,
    init_FN: t.Tensor,
    gram_FF: t.Tensor | None = None,
    steps: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 1024,
    eval_every: int = 50,
    patience: int = 20,
    seed: int = 0,
    device: t.device | str = "cpu",
) -> TransportFit:
    """Fit ``A`` (``[F, N]``) so that ``A delta ~= y``: Adam on the mean squared error
    (in the ``gram_FF`` norm when given: the logit-space loss), cosine learning-rate
    decay to zero over ``steps``, no weight decay. Every ``eval_every`` steps the
    held-out relative error is measured; the best ``A`` is kept, and training stops
    after ``patience`` evaluations without improvement."""
    if not heldout_Bool_P.any() or heldout_Bool_P.all():
        raise ValueError("need both training and held-out pairs")
    delta_PN, y_PF = delta_PN.to(device).float(), y_PF.to(device).float()
    train_delta_PN, train_y_PF = delta_PN[~heldout_Bool_P.to(device)], y_PF[~heldout_Bool_P.to(device)]
    heldout_delta_PN, heldout_y_PF = delta_PN[heldout_Bool_P.to(device)], y_PF[heldout_Bool_P.to(device)]
    gram_FF = gram_FF.to(device).float() if gram_FF is not None else None
    heldout_target_norm = float(_squared_error_P(heldout_y_PF, gram_FF).sum())
    # The training loss is normalised by the mean target norm so its scale is O(1) in the log;
    # Adam's steps do not depend on it.
    loss_scale = float(_squared_error_P(train_y_PF, gram_FF).mean())

    matrix_FN = init_FN.to(device).float().clone().requires_grad_(True)
    optimizer = t.optim.Adam([matrix_FN], lr=lr)
    scheduler = t.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))
    )
    generator = t.Generator(device=train_delta_PN.device).manual_seed(seed)

    def heldout_relative_error() -> float:
        with t.no_grad():
            residual_PF = heldout_delta_PN @ matrix_FN.T - heldout_y_PF
            return float(_squared_error_P(residual_PF, gram_FF).sum()) / heldout_target_norm

    with t.no_grad():
        # Least squares on the training pairs in fp64, with a tiny ridge for rank-deficient
        # designs: A^T = (D^T D + lambda I)^-1 D^T Y.
        gram_NN = train_delta_PN.double().T @ train_delta_PN.double()
        ridge = 1e-8 * float(gram_NN.diagonal().mean())
        gram_NN += ridge * t.eye(gram_NN.shape[0], dtype=t.float64, device=gram_NN.device)
        lstsq_FN = t.linalg.solve(gram_NN, train_delta_PN.double().T @ train_y_PF.double()).T.float()
        lstsq_residual_PF = heldout_delta_PN @ lstsq_FN.T - heldout_y_PF
        lstsq_error = float(_squared_error_P(lstsq_residual_PF, gram_FF).sum()) / heldout_target_norm

    init_error = heldout_relative_error()
    best_error, best_step, best_FN = init_error, 0, matrix_FN.detach().clone()
    history: list[dict[str, float]] = [{"step": 0, "heldout_relative_error": init_error}]
    evals_without_improvement = 0
    running_loss = 0.0
    for step in range(1, steps + 1):
        batch_Int = t.randint(len(train_delta_PN), (batch_size,), generator=generator, device=train_delta_PN.device)
        residual_PF = train_delta_PN[batch_Int] @ matrix_FN.T - train_y_PF[batch_Int]
        loss = _squared_error_P(residual_PF, gram_FF).mean() / loss_scale
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += float(loss.detach())
        if step % eval_every == 0 or step == steps:
            error = heldout_relative_error()
            history.append(
                {"step": step, "train_loss": running_loss / eval_every, "heldout_relative_error": error}
            )
            running_loss = 0.0
            if error < best_error:
                best_error, best_step, best_FN = error, step, matrix_FN.detach().clone()
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience:
                    break
    return TransportFit(
        matrix_FN=best_FN.cpu(),
        best_step=best_step,
        init_heldout_relative_error=init_error,
        best_heldout_relative_error=best_error,
        history=history,
        lstsq_heldout_relative_error=lstsq_error,
    )


@dataclass
class PairsMeta:
    """What a pairs directory was collected with (``meta.json``): enough to build the
    fitted lens's config without loading the model."""

    hf_model_name: str
    d_model: int
    layers: list[int]
    relative_end_transport_layer: int
    max_seq_len: int
    skip_first_n_positions: int
    num_prompts: int
    family: str
    pairs_per_prompt: int
    seed: int

    def save(self, pairs_dir: str | Path) -> None:
        (Path(pairs_dir) / "meta.json").write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, pairs_dir: str | Path) -> PairsMeta:
        return cls(**json.loads((Path(pairs_dir) / "meta.json").read_text()))


def intervention_lens_config(meta: PairsMeta, *, checkpoint_name: str, num_prompts_trained_on: int) -> LensConfig:
    """The config of a lens fitted on pairs: a plain Jacobian-format lens (``lrp_mode``
    ``"none"``) with the pairs' layer geometry and position settings."""
    return LensConfig(
        hf_model_name=meta.hf_model_name,
        checkpoint_name=checkpoint_name,
        lens_type="jacobian",
        checkpoint_every_n_prompts=None,
        source_layers=list(meta.layers),
        relative_end_transport_layer=meta.relative_end_transport_layer,
        d_model=meta.d_model,
        num_prompts_trained_on=num_prompts_trained_on,
        max_seq_len=meta.max_seq_len,
        skip_first_n_positions=meta.skip_first_n_positions,
    )


def _check_pairs_metas_agree(metas: Sequence[PairsMeta]) -> None:
    """Pair directories fitted together must come from the same model, prompts and
    position settings (their families and pair counts may differ)."""
    for meta in metas[1:]:
        for field_name in (
            "hf_model_name",
            "d_model",
            "relative_end_transport_layer",
            "max_seq_len",
            "skip_first_n_positions",
            "num_prompts",
        ):
            ours, theirs = getattr(metas[0], field_name), getattr(meta, field_name)
            if ours != theirs:
                raise ValueError(
                    f"pairs of {metas[0].family} and {meta.family} disagree on {field_name}: "
                    f"{ours!r} != {theirs!r}"
                )


def fit_intervention_lens(
    pairs_dirs: str | Path | Sequence[str | Path],
    *,
    layers: Sequence[int],
    target: TargetKind = "sum_future",
    loss: LossKind = "residual",
    init_lens: JacobianLens | None = None,
    checkpoint_name: str,
    holdout_fraction: float = 0.1,
    holdout_seed: int = 0,
    device: t.device | str = "cpu",
    **fit_kwargs: object,
) -> tuple[JacobianLens, dict[int, TransportFit]]:
    """:func:`fit_transport` at every layer of ``layers`` on the pairs in ``pairs_dirs``
    (one family's directory from the ``collect-pairs`` stage, or several fitted together
    as one pooled set of pairs), each from the identity or from ``init_lens``'s map, with
    the same held-out prompts at every layer. The logit loss reads ``gram.pt`` from the
    first pairs directory's parent (:func:`unembed_gram`)."""
    if isinstance(pairs_dirs, (str, Path)):
        pairs_dirs = [pairs_dirs]
    metas = [PairsMeta.load(pairs_dir) for pairs_dir in pairs_dirs]
    _check_pairs_metas_agree(metas)
    meta = metas[0]
    heldout_prompts_Bool = heldout_prompt_mask(meta.num_prompts, holdout_fraction, seed=holdout_seed)
    gram_FF = None
    if loss == "logit":
        gram_FF = t.load(Path(pairs_dirs[0]).parent / "gram.pt", map_location="cpu", weights_only=True)
    fits_L_dict: dict[int, TransportFit] = {}
    for layer in layers:
        pairs = LayerPairs.concatenate(
            [LayerPairs.load(pairs_path(pairs_dir, layer)) for pairs_dir in pairs_dirs]
        )
        init_FN = (
            init_lens.jacobians_L_dict_FN[layer]  # type: ignore[index]
            if init_lens is not None
            else t.eye(meta.d_model)
        )
        layer_start_time = time.perf_counter()
        fit = fit_transport(
            pairs.delta_PN,
            pairs.target(target),
            heldout_prompts_Bool[pairs.prompt_idx_Int_P],
            init_FN=init_FN,
            gram_FF=gram_FF,
            device=device,
            **fit_kwargs,  # type: ignore[arg-type]
        )
        fits_L_dict[layer] = fit
        logger.info(
            "  layer %d: %d pairs, held-out relative error %.4f (init) -> %.4f (step %d), "
            "least squares %.4f  %.0fs",
            layer,
            len(pairs.delta_PN),
            fit.init_heldout_relative_error,
            fit.best_heldout_relative_error,
            fit.best_step,
            fit.lstsq_heldout_relative_error,
            time.perf_counter() - layer_start_time,
        )
    num_training_prompts = int((~heldout_prompts_Bool).sum())
    lens = JacobianLens(
        jacobians={layer: fit.matrix_FN for layer, fit in fits_L_dict.items()},
        config=intervention_lens_config(
            meta, checkpoint_name=checkpoint_name, num_prompts_trained_on=num_training_prompts
        ),
    )
    return lens, fits_L_dict


### THE TUNED-LENS-STYLE BASELINE


def _kl_from_model(model: LensModel, lens_input_PF: t.Tensor, model_logprobs_PV: t.Tensor) -> t.Tensor:
    """``KL(p_model || p_lens)`` per row, ``p_lens = softmax(unembed(lens_input))``."""
    lens_logprobs_PV = t.log_softmax(model.unembed(lens_input_PF).float(), dim=-1)
    return (model_logprobs_PV.exp() * (model_logprobs_PV - lens_logprobs_PV)).sum(dim=-1)


def fit_output_kl(
    model: LensModel,
    pool: CleanPool,
    *,
    layer: int,
    heldout_prompts_Bool: t.Tensor,
    init_FN: t.Tensor,
    steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    eval_every: int = 100,
    eval_rows: int = 2048,
    patience: int = 10,
    seed: int = 0,
) -> TransportFit:
    """The Tuned-Lens-style fit of one layer: ``A`` such that ``unembed(A h)`` matches
    the model's own next-token distribution ``unembed(h_target)`` at the same position,
    by Adam on ``KL(p_model || p_lens)`` over the pool's kept positions, cosine decay,
    early stopping on a fixed sample of ``eval_rows`` held-out positions. Linear (no
    bias), so it saves as a Jacobian-format lens. Errors in the result are mean
    held-out KLs, not relative errors."""
    device = getattr(model, "input_device", t.device("cpu"))
    heldout_Bool_R = heldout_prompts_Bool[pool.prompt_idx_Int_R]
    sources_RN = pool.sources_L_dict_RN[layer]
    train_rows_Int = t.where(~heldout_Bool_R)[0]
    heldout_rows_Int = t.where(heldout_Bool_R)[0]
    generator = t.Generator().manual_seed(seed)
    heldout_rows_Int = heldout_rows_Int[t.randperm(len(heldout_rows_Int), generator=generator)[:eval_rows]]

    def model_logprobs(rows_Int: t.Tensor) -> t.Tensor:
        with t.no_grad():
            return t.log_softmax(model.unembed(pool.target_RF[rows_Int].to(device).float()).float(), dim=-1)

    matrix_FN = init_FN.to(device).float().clone().requires_grad_(True)
    optimizer = t.optim.Adam([matrix_FN], lr=lr)
    scheduler = t.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))
    )

    def heldout_kl() -> float:
        total = 0.0
        with t.no_grad():
            for start in range(0, len(heldout_rows_Int), batch_size):
                rows_Int = heldout_rows_Int[start : start + batch_size]
                lens_input_PF = sources_RN[rows_Int].to(device).float() @ matrix_FN.T
                total += float(_kl_from_model(model, lens_input_PF, model_logprobs(rows_Int)).sum())
        return total / len(heldout_rows_Int)

    init_kl = heldout_kl()
    best_kl, best_step, best_FN = init_kl, 0, matrix_FN.detach().clone()
    history: list[dict[str, float]] = [{"step": 0, "heldout_kl": init_kl}]
    evals_without_improvement = 0
    running_loss = 0.0
    for step in range(1, steps + 1):
        rows_Int = train_rows_Int[t.randint(len(train_rows_Int), (batch_size,), generator=generator)]
        lens_input_PF = sources_RN[rows_Int].to(device).float() @ matrix_FN.T
        loss = _kl_from_model(model, lens_input_PF, model_logprobs(rows_Int)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += float(loss.detach())
        if step % eval_every == 0 or step == steps:
            kl = heldout_kl()
            history.append({"step": step, "train_kl": running_loss / eval_every, "heldout_kl": kl})
            running_loss = 0.0
            if kl < best_kl:
                best_kl, best_step, best_FN = kl, step, matrix_FN.detach().clone()
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience:
                    break
    return TransportFit(
        matrix_FN=best_FN.cpu(),
        best_step=best_step,
        init_heldout_relative_error=init_kl,
        best_heldout_relative_error=best_kl,
        history=history,
    )
