"""Fitting lens maps by gradient descent on a swap loss.

The probe-swap eval (:mod:`lens_evals.causal_evals`) edits the residual along a lens's
per-token directions. With the unit lens vectors ``u_X, u_Y`` of two tokens
(``u_tok`` proportional to ``W_U[tok] @ A_l``, the J-lens vector of :func:`jlens_vectors`
with ``A_l`` in place of ``J_l``), it pins the coordinates of ``h`` on ``[u_X, u_Y]`` to the
clean pass's coordinates exchanged (times a scale), at every band layer and every position
but 0, and asks whether the model's output moves as if the concept were ``Y``. This module
trains the maps ``A_l`` of every band layer jointly on that same edit, on unlabelled text,
by backpropagating through the model:

- ``next_token``: ``X`` is the model's own top next-token prediction at a position ``s``,
  ``Y`` an alternative from its top ``k``; the loss is the cross-entropy of ``Y`` at ``s``
  under the edit.
- ``counterfactual``: ``X`` is the token at position ``s`` of the text, ``Y`` an alternative
  the model found plausible for that slot (its top ``k`` at ``s - 1``); the target is the
  model's own next-token distribution after literally writing ``Y`` at ``s``, and the loss
  is its KL to the edited run at positions ``s .. s + window - 1``. This is the shape of
  the eval's main trial: swap a concept already in context and expect what follows to
  change accordingly.
- ``supervised``: rows from a built dataset of templated prompts (``"... {X} ..."``) with
  known answers; ``Y`` is another word of the same template and the loss is the
  cross-entropy of ``Y``'s answer (any of its single-token surfaces) at the final position,
  over word tokens only (the formatting tokens are left out, as in probe-swap's word rank).

Shapes: ``B`` batch rows (one swap example each), ``S`` positions, ``N`` / ``F`` source /
target width, ``V`` vocabulary, ``W`` the counterfactual window.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch as t
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel
from workspace_lens.config import LensConfig
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.residual_streams import has_residual_streams, residual_mean_over_streams
from workspace_lens.utils import (
    get_position_mask_with_early_skips,
    get_unembed_matrix,
    model_device_of,
)

logger = logging.getLogger(__name__)

SwapMode = Literal["next_token", "counterfactual", "supervised"]


### THE EDIT


def swap_edit(
    residual_BSN: t.Tensor,
    clean_residual_BSN: t.Tensor,
    source_vectors_BN: t.Tensor,
    target_vectors_BN: t.Tensor,
    *,
    scale: float,
    edit_mask_Bool_S: t.Tensor,
) -> t.Tensor:
    """The probe-swap clamp of every row, differentiably: with the unit vectors
    ``V = [u_X, u_Y]`` of each row, ``h <- h + V (c* - c)`` at the masked positions,
    ``c = pinv(V) h`` the current coordinates and ``c*`` the clean pass's coordinates
    exchanged and times ``scale`` (as
    :func:`lens_evals.causal_evals.runner._swapped_coordinate_clamp` and
    :class:`~workspace_lens.interventions.interventions.Clamp` do it). The products with the
    ``[B, S, N]`` residuals run in the residual's dtype (the result has it too), the rest in
    fp32; the vectors need not be normalised (they are here)."""
    dtype = residual_BSN.dtype
    unit_source_BN = source_vectors_BN / source_vectors_BN.norm(dim=-1, keepdim=True)
    unit_target_BN = target_vectors_BN / target_vectors_BN.norm(dim=-1, keepdim=True)
    basis_BN2 = t.stack([unit_source_BN, unit_target_BN], dim=-1).float()
    # pinv(V) = (V^T V)^-1 V^T; the tiny ridge only matters for near-parallel pairs, which the
    # eval skips.
    gram_B22 = basis_BN2.transpose(-1, -2) @ basis_BN2 + 1e-6 * t.eye(2, device=basis_BN2.device)
    pinv_BN2 = t.linalg.solve(gram_B22, basis_BN2.transpose(-1, -2)).transpose(-1, -2).to(dtype)
    coordinates_BS2 = residual_BSN @ pinv_BN2
    targets_BS2 = scale * (clean_residual_BSN.to(dtype) @ pinv_BN2).flip(-1)
    # The mask is applied to the [B, S, 2] shift, and the add is fused into the product.
    shift_BS2 = (targets_BS2.float() - coordinates_BS2.float()) * edit_mask_Bool_S[None, :, None]
    return t.baddbmm(residual_BSN, shift_BS2.to(dtype), basis_BN2.transpose(-1, -2).to(dtype))


class SwapEditHooks:
    """Writer forward hooks applying :func:`swap_edit` at every band layer, with the
    per-row vectors ``vectors_fn(layer) -> (u_X [B, N], u_Y [B, N])`` and the clean
    residuals of each band layer. The edit is computed in ``edit_dtype`` (default fp32; None:
    the residual's own dtype) and cast back to the residual's dtype; gradients flow to whatever
    the vectors depend on."""

    def __init__(
        self,
        model: LensModel,
        *,
        vectors_fn: Callable[[int], tuple[t.Tensor, t.Tensor]],
        clean_residuals_L_dict_BSN: dict[int, t.Tensor],
        scale: float,
        edit_mask_Bool_S: t.Tensor,
        edit_dtype: t.dtype | None = t.float32,
    ) -> None:
        self.model = model
        self.vectors_fn = vectors_fn
        self.clean_residuals_L_dict_BSN = clean_residuals_L_dict_BSN
        self.scale = scale
        self.edit_mask_Bool_S = edit_mask_Bool_S
        self.edit_dtype = edit_dtype
        self._handles: list[t.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int) -> Callable[..., t.Tensor | tuple]:
        def hook(module: nn.Module, inputs: tuple, output: t.Tensor | tuple) -> t.Tensor | tuple:
            block_output_BSN = output if t.is_tensor(output) else output[0]
            if has_residual_streams(block_output_BSN):
                raise NotImplementedError("swap edits do not lift to a multi-stream residual")
            source_vectors_BN, target_vectors_BN = self.vectors_fn(layer)
            edited_BSN = swap_edit(
                block_output_BSN.to(self.edit_dtype or block_output_BSN.dtype),
                self.clean_residuals_L_dict_BSN[layer],
                source_vectors_BN,
                target_vectors_BN,
                scale=self.scale,
                edit_mask_Bool_S=self.edit_mask_Bool_S,
            ).to(block_output_BSN.dtype)
            return edited_BSN if t.is_tensor(output) else (edited_BSN, *output[1:])

        return hook

    def __enter__(self) -> SwapEditHooks:
        try:
            for layer in self.clean_residuals_L_dict_BSN:
                self._handles.append(self.model.layers[layer].register_forward_hook(self._make_hook(layer)))
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


### EXAMPLES


@dataclass(frozen=True)
class SwapExample:
    """One training row: swap tokens ``source_id`` (X) and ``target_id`` (Y) around
    position ``position`` (s) of prompt ``prompt_idx``; in supervised mode the answer's
    token ids too."""

    prompt_idx: int
    position: int
    source_id: int
    target_id: int
    answer_ids: tuple[int, ...] = ()


@t.no_grad()
def _final_logits(model: LensModel, input_ids_Int_BS: t.Tensor) -> t.Tensor:
    """The model's own logits ``[B, S, V]`` (fp32) on ``input_ids``."""
    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids_Int_BS)
        final_BSF = residual_mean_over_streams(recorder.activations[final_layer])
    return model.unembed(final_BSF.float()).float()


def full_length_prompts(model: LensModel, prompts: Sequence[str], max_seq_len: int) -> list[int]:
    """Indices of the prompts that fill ``max_seq_len`` tokens, so rows from different
    prompts batch without padding."""
    return [idx for idx, prompt in enumerate(prompts) if model.encode(prompt, max_length=max_seq_len).shape[1] == max_seq_len]


def draw_swap_examples(
    model: LensModel,
    prompts: Sequence[str],
    prompt_indices: Sequence[int],
    *,
    mode: SwapMode,
    examples_per_prompt: int,
    top_k: int,
    excluded_mask_Bool_V: t.Tensor,
    max_seq_len: int,
    skip_first_n_positions: int,
    window: int,
    seed: int = 0,
) -> list[SwapExample]:
    """Up to ``examples_per_prompt`` examples per prompt at distinct kept positions
    (:func:`get_position_mask_with_early_skips`; in counterfactual mode also leaving
    ``window`` predictions after ``s``). Positions whose ``X`` (the model's top prediction,
    or the token at ``s``) is an excluded id (pass the formatting tokens) are skipped, and
    ``Y`` is drawn uniformly from the top ``top_k`` candidates that are neither excluded
    nor ``X``. One clean forward per prompt."""
    generator = t.Generator().manual_seed(seed)
    examples: list[SwapExample] = []
    for prompt_idx in prompt_indices:
        input_ids_Int_1S = model.encode(prompts[prompt_idx], max_length=max_seq_len)
        seq_len = input_ids_Int_1S.shape[1]
        raw_logits_SV = _final_logits(model, input_ids_Int_1S)[0].cpu()
        logits_SV = raw_logits_SV.masked_fill(excluded_mask_Bool_V, float("-inf"))
        kept_Bool_S = get_position_mask_with_early_skips(seq_len, skip_first_n_positions)
        if mode == "counterfactual":
            kept_Bool_S[seq_len - window + 1 :] = False
        positions_Int = t.where(kept_Bool_S)[0]
        positions_Int = positions_Int[t.randperm(len(positions_Int), generator=generator)]
        count = 0
        for position in positions_Int.tolist():
            if count == examples_per_prompt:
                break
            if mode == "next_token":
                source_id = int(raw_logits_SV[position].argmax())
                candidates_Int = logits_SV[position].topk(top_k + 1).indices
            else:
                source_id = int(input_ids_Int_1S[0, position])
                candidates_Int = logits_SV[position - 1].topk(top_k + 1).indices
            if excluded_mask_Bool_V[source_id]:
                continue
            alternatives = [int(i) for i in candidates_Int if int(i) != source_id][:top_k]
            target_id = alternatives[int(t.randint(len(alternatives), (1,), generator=generator))]
            examples.append(SwapExample(prompt_idx, position, source_id, target_id))
            count += 1
    return examples


### THE LOSS


def prompt_ids(
    model: LensModel, prompts: Sequence[str], prompt_idx: int, *, max_seq_len: int, cache: dict[int, t.Tensor]
) -> t.Tensor:
    """Prompt ``prompt_idx``'s token ids ``[S]`` (CPU), tokenised once into ``cache``."""
    if prompt_idx not in cache:
        cache[prompt_idx] = model.encode(prompts[prompt_idx], max_length=max_seq_len)[0].cpu()
    return cache[prompt_idx]


@dataclass
class SwapBatch:
    """A batch of examples on the device: input ids ``[B, S]``, positions ``[B]``, X and Y ids
    ``[B]``, and the answer ids ``[B, A]`` padded with -1. Rows of different lengths are padded
    on the right: the model is causal, so no position sees the padding after it, and every
    read position is a real token (the swap edits the padding too, which nothing reads)."""

    input_ids_Int_BS: t.Tensor
    position_Int_B: t.Tensor
    source_Int_B: t.Tensor
    target_Int_B: t.Tensor
    answers_Int_BA: t.Tensor
    prompt_Int_B: t.Tensor | None = None  # each row's prompt index (for a CleanResidualCache)

    @classmethod
    def build(
        cls,
        model: LensModel,
        prompts: Sequence[str],
        examples: Sequence[SwapExample],
        *,
        max_seq_len: int,
        ids_cache: dict[int, t.Tensor] | None = None,
    ) -> SwapBatch:
        """``ids_cache`` (prompt index -> token ids) saves re-tokenising across batches."""
        device = model_device_of(model)
        cache = ids_cache if ids_cache is not None else {}
        rows = [prompt_ids(model, prompts, example.prompt_idx, max_seq_len=max_seq_len, cache=cache) for example in examples]
        pad_id = getattr(model.tokenizer, "pad_token_id", None) or 0
        input_ids_Int_BS = t.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_id).to(device)
        as_tensor = lambda values: t.tensor(values, dtype=t.long, device=device)  # noqa: E731
        width = max(1, max(len(example.answer_ids) for example in examples))
        return cls(
            input_ids_Int_BS=input_ids_Int_BS,
            position_Int_B=as_tensor([example.position for example in examples]),
            source_Int_B=as_tensor([example.source_id for example in examples]),
            target_Int_B=as_tensor([example.target_id for example in examples]),
            answers_Int_BA=as_tensor(
                [list(example.answer_ids) + [-1] * (width - len(example.answer_ids)) for example in examples]
            ),
            prompt_Int_B=as_tensor([example.prompt_idx for example in examples]),
        )


class CleanResidualCache:
    """The clean residuals of a set of prompts at ``layers``, from one no-grad pass over them,
    kept on the model's device in the residual dtype: ``[L, T + 1, N]`` for ``T`` prompt tokens
    in all, the last row zeros (read at padding). A swap row's clean pass depends on its prompt
    only, so the training steps read it here instead of running it again."""

    def __init__(
        self,
        model: LensModel,
        prompts: Sequence[str],
        prompt_indices: Sequence[int],
        *,
        layers: Sequence[int],
        max_seq_len: int,
        ids_cache: dict[int, t.Tensor],
        batch_size: int = 256,
    ) -> None:
        device = model_device_of(model)
        self.layers = sorted(layers)
        self._row = {layer: row for row, layer in enumerate(self.layers)}
        ids = {idx: prompt_ids(model, prompts, idx, max_seq_len=max_seq_len, cache=ids_cache) for idx in set(prompt_indices)}
        order = sorted(ids, key=lambda idx: len(ids[idx]))  # little padding per pass
        lengths_P = t.tensor([len(ids[idx]) for idx in order])
        offsets_P = t.cumsum(lengths_P, 0) - lengths_P
        self.offset_P = t.full((len(prompts),), -1, dtype=t.long, device=device)
        self.length_P = t.zeros(len(prompts), dtype=t.long, device=device)
        self.offset_P[order] = offsets_P.to(device)
        self.length_P[order] = lengths_P.to(device)
        num_tokens = int(lengths_P.sum())
        self.store_LTN: t.Tensor | None = None
        pad_id = getattr(model.tokenizer, "pad_token_id", None) or 0
        with t.no_grad():
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                input_ids_Int_BS = t.nn.utils.rnn.pad_sequence(
                    [ids[idx] for idx in chunk], batch_first=True, padding_value=pad_id
                ).to(device)
                real_Bool_BS = t.arange(input_ids_Int_BS.shape[1], device=device) < self.length_P[chunk][:, None]
                first = int(offsets_P[start])
                with ActivationRecorder(model.layers, at=self.layers) as recorder:
                    model.forward(input_ids_Int_BS)
                    for layer in self.layers:
                        residual_BSN = residual_mean_over_streams(recorder.activations[layer])
                        if self.store_LTN is None:
                            self.store_LTN = t.zeros(
                                len(self.layers), num_tokens + 1, residual_BSN.shape[-1], dtype=residual_BSN.dtype, device=device
                            )
                        # The chunk's prompts are consecutive in the store, row-major like the mask.
                        tokens_TN = residual_BSN[real_Bool_BS]
                        self.store_LTN[self._row[layer], first : first + len(tokens_TN)] = tokens_TN

    @staticmethod
    def size_in_bytes(model: LensModel, num_tokens: int, num_layers: int) -> int:
        """The store's size for ``num_tokens`` prompt tokens at ``num_layers`` layers."""
        unembed_VF = get_unembed_matrix(model)
        return (num_tokens + 1) * num_layers * unembed_VF.shape[-1] * unembed_VF.element_size()

    def residuals(self, batch: SwapBatch, layers: Sequence[int]) -> dict[int, t.Tensor]:
        """``{layer: [B, S, N]}`` for the batch's prompts, zeros past each prompt's end."""
        if batch.prompt_Int_B is None or bool((self.offset_P[batch.prompt_Int_B] < 0).any()):
            raise ValueError("the batch has prompts this cache was not built for")
        assert self.store_LTN is not None
        positions_Int_S = t.arange(batch.input_ids_Int_BS.shape[1], device=self.offset_P.device)
        token_Int_BS = self.offset_P[batch.prompt_Int_B][:, None] + positions_Int_S
        real_Bool_BS = positions_Int_S < self.length_P[batch.prompt_Int_B][:, None]
        token_Int_BS = t.where(real_Bool_BS, token_Int_BS, self.store_LTN.shape[1] - 1)
        return {layer: self.store_LTN[self._row[layer]][token_Int_BS] for layer in layers}


class SwapLoss:
    """The swap loss of a set of maps ``{layer: A_l [F, N]}`` on a :class:`SwapBatch`:
    clean forward (band residuals, no grad), in counterfactual mode the counterfactual
    forward (no grad), then the edited forward with :class:`SwapEditHooks` at every band
    layer, whose vectors ``W_U[tok] @ A_l`` carry the gradient back to the maps."""

    def __init__(
        self,
        model: LensModel,
        *,
        layers: Sequence[int],
        mode: SwapMode,
        scale: float,
        window: int,
        excluded_mask_Bool_V: t.Tensor | None = None,
        edit_dtype: t.dtype | None = t.float32,
        clean_cache: CleanResidualCache | None = None,
    ) -> None:
        self.model = model
        self.layers = sorted(layers)
        self.mode = mode
        self.scale = scale
        self.window = window
        self.edit_dtype = edit_dtype  # see SwapEditHooks
        self.clean_cache = clean_cache  # read instead of running the clean pass
        # Supervised mode scores word tokens only: these ids are dropped from the distribution.
        self.excluded_mask_Bool_V = (
            excluded_mask_Bool_V.to(model_device_of(model)) if excluded_mask_Bool_V is not None else None
        )
        self.unembed_VF = get_unembed_matrix(model)
        self.final_layer = model.n_layers - 1

    def _window_positions_BW(self, batch: SwapBatch) -> t.Tensor:
        """The positions the loss reads: ``s`` (next_token) or ``s .. s + window - 1``."""
        width = self.window if self.mode == "counterfactual" else 1
        return batch.position_Int_B[:, None] + t.arange(width, device=batch.position_Int_B.device)

    @t.no_grad()
    def _targets(self, batch: SwapBatch, layers: Sequence[int]) -> tuple[dict[int, t.Tensor], t.Tensor]:
        """Clean residuals at the edited ``layers`` ``{layer: [B, S, N]}`` (the model's dtype;
        from the clean cache when there is one) and the target: Y's ids ``[B, 1]``
        (next_token), the answer ids ``[B, A]`` (supervised) or the counterfactual log-probs
        ``[B, W, V]``."""
        if self.clean_cache is not None:
            clean_L_dict_BSN = self.clean_cache.residuals(batch, layers)
        else:
            with ActivationRecorder(self.model.layers, at=layers) as recorder:
                self.model.forward(batch.input_ids_Int_BS)
                clean_L_dict_BSN = {layer: residual_mean_over_streams(recorder.activations[layer]) for layer in layers}
        if self.mode == "next_token":
            return clean_L_dict_BSN, batch.target_Int_B[:, None]
        if self.mode == "supervised":
            return clean_L_dict_BSN, batch.answers_Int_BA
        rows_Int_B = t.arange(len(batch.position_Int_B), device=batch.position_Int_B.device)
        counterfactual_ids_Int_BS = batch.input_ids_Int_BS.clone()
        counterfactual_ids_Int_BS[rows_Int_B, batch.position_Int_B] = batch.target_Int_B
        logits_BSV = _final_logits(self.model, counterfactual_ids_Int_BS)
        window_logits_BWV = logits_BSV[rows_Int_B[:, None], self._window_positions_BW(batch)]
        return clean_L_dict_BSN, t.log_softmax(window_logits_BWV, dim=-1)

    def __call__(
        self,
        maps_L_dict_FN: dict[int, t.Tensor],
        batch: SwapBatch,
        *,
        active_layers: Sequence[int] | None = None,
    ) -> tuple[t.Tensor, t.Tensor]:
        """``(loss, hits_B)``: the mean loss (differentiable in the maps) and per row
        whether the edited run's top token at ``s`` is the target's (Y, the answer, or the
        counterfactual run's top token). The swap is made at ``active_layers`` only (default:
        every layer of the loss); the other layers are left alone."""
        layers = sorted(active_layers) if active_layers is not None else self.layers
        clean_L_dict_BSN, target = self._targets(batch, layers)
        source_rows_BF = self.unembed_VF[batch.source_Int_B].float()
        target_rows_BF = self.unembed_VF[batch.target_Int_B].float()

        def vectors_fn(layer: int) -> tuple[t.Tensor, t.Tensor]:
            matrix_FN = maps_L_dict_FN[layer]
            return source_rows_BF @ matrix_FN, target_rows_BF @ matrix_FN

        seq_len = batch.input_ids_Int_BS.shape[1]
        edit_mask_Bool_S = t.arange(seq_len, device=batch.input_ids_Int_BS.device) > 0  # all but the sink
        hooks = SwapEditHooks(
            self.model,
            vectors_fn=vectors_fn,
            clean_residuals_L_dict_BSN=clean_L_dict_BSN,
            scale=self.scale,
            edit_mask_Bool_S=edit_mask_Bool_S,
            edit_dtype=self.edit_dtype,
        )
        # Differentiable when grad is enabled (training), not under no_grad (held-out evals).
        with hooks, ActivationRecorder(self.model.layers, at=[self.final_layer]) as recorder:
            self.model.forward(batch.input_ids_Int_BS)
            final_BSF = residual_mean_over_streams(recorder.activations[self.final_layer])
            rows_Int_B = t.arange(len(batch.position_Int_B), device=final_BSF.device)
            logits_BWV = self.model.unembed(final_BSF[rows_Int_B[:, None], self._window_positions_BW(batch)]).float()
            if self.mode == "supervised" and self.excluded_mask_Bool_V is not None:
                logits_BWV = logits_BWV.masked_fill(self.excluded_mask_Bool_V, float("-inf"))
            logprobs_BWV = t.log_softmax(logits_BWV, dim=-1)
            if self.mode == "supervised":
                valid_Bool_BA = target >= 0
                answer_logprobs_BA = logprobs_BWV[rows_Int_B[:, None], 0, target.clamp(min=0)]
                answer_logprobs_BA = answer_logprobs_BA.masked_fill(~valid_Bool_BA, float("-inf"))
                loss = -t.logsumexp(answer_logprobs_BA, dim=-1).mean()
                top_Int_B = logits_BWV[:, 0].argmax(dim=-1)
                hits_B = ((target == top_Int_B[:, None]) & valid_Bool_BA).any(dim=-1)
            elif self.mode == "next_token":
                loss = -logprobs_BWV[rows_Int_B, 0, target[:, 0]].mean()
                hits_B = logits_BWV[:, 0].argmax(dim=-1) == target[:, 0]
            else:
                loss = (target.exp() * (target - logprobs_BWV)).sum(dim=-1).mean()
                hits_B = logits_BWV[:, 0].argmax(dim=-1) == target[:, 0].argmax(dim=-1)
        return loss, hits_B


### TRAINING


@dataclass
class SwapFit:
    """The trained maps, the held-out loss and hit rate at the init and at the kept
    (best) step, and the evaluation history."""

    maps_L_dict_FN: dict[int, t.Tensor]
    best_step: int
    init_heldout_loss: float
    best_heldout_loss: float
    init_heldout_hit_rate: float
    best_heldout_hit_rate: float
    history: list[dict[str, float]]


def fit_swap_maps(
    model: LensModel,
    prompts: Sequence[str],
    train_examples: Sequence[SwapExample],
    heldout_examples: Sequence[SwapExample],
    *,
    init_maps_L_dict_FN: dict[int, t.Tensor],
    mode: SwapMode,
    scale: float = 1.0,
    window: int = 8,
    max_seq_len: int = 128,
    steps: int = 2000,
    batch_size: int = 16,
    relative_lr: float = 1e-2,
    eval_every: int = 100,
    patience: int = 5,
    seed: int = 0,
    excluded_mask_Bool_V: t.Tensor | None = None,
    monitor_examples: Sequence[SwapExample] = (),
    window_sizes: Sequence[int] = (),
    all_layers_probability: float = 1.0,
    log_fn: Callable[[dict[str, float]], None] | None = None,
    log_every: int = 10,
    length_pool: int = 1,
    edit_dtype: t.dtype | None = t.float32,
    clean_cache_gb: float = 0.0,
) -> SwapFit:
    """Adam on the swap loss over every band layer's map jointly, each map's learning
    rate ``relative_lr`` times the RMS of its initial entries (the loss ignores each
    map's overall scale, so the step is set relative to it), cosine decay over
    ``steps``. Every ``eval_every`` steps the held-out loss is measured; the best maps
    are kept and training stops after ``patience`` evaluations without improvement. Each
    batch is drawn uniformly from the training rows, padded to one length
    (:class:`SwapBatch`). With ``length_pool`` > 1 the rows of that many batches are drawn at
    once, sorted by prompt length and cut into batches served in random order: the same rows,
    grouped so each batch has little padding. With ``clean_cache_gb`` > 0 the clean pass of
    every prompt is run once up front and kept on the GPU (:class:`CleanResidualCache`) when it
    takes at most that many GB and fits next to training (:func:`_clean_cache_within`);
    otherwise each step runs it. ``edit_dtype`` is the swap edit's (:class:`SwapEditHooks`). Each step edits every layer with
    probability ``all_layers_probability``; otherwise one random window of consecutive layers
    (a size drawn uniformly from ``window_sizes``, positions in the sorted layer list), with
    the other layers left alone, so each map must also work on its own. Held-out evaluations
    always edit every layer. ``log_fn`` (e.g. a W&B run's ``log``) gets the step's training loss,
    hit rate, number of swapped layers and learning rate every ``log_every`` steps, and every
    evaluation's metrics, each as a dict with a ``step`` key. ``monitor_examples`` (e.g. swaps from
    held-out relations) are scored at every evaluation and logged in the history as
    ``monitor_loss`` / ``monitor_hit_rate``, but never used for stopping or picking the maps."""
    device = model_device_of(model)
    swap_loss = SwapLoss(
        model,
        layers=list(init_maps_L_dict_FN),
        mode=mode,
        scale=scale,
        window=window,
        excluded_mask_Bool_V=excluded_mask_Bool_V,
        edit_dtype=edit_dtype,
    )
    ids_cache: dict[int, t.Tensor] = {}

    def prompt_length(example: SwapExample) -> int:
        return len(prompt_ids(model, prompts, example.prompt_idx, max_seq_len=max_seq_len, cache=ids_cache))

    def chunks(examples: Sequence[SwapExample]) -> list[list[SwapExample]]:
        # Rows are scored independently, so the evaluation sets are grouped by length too.
        examples = sorted(examples, key=prompt_length)
        return [list(examples[start : start + batch_size]) for start in range(0, len(examples), batch_size)]

    heldout_chunks, monitor_chunks = chunks(heldout_examples), chunks(monitor_examples)
    maps_L_dict_FN = {
        layer: matrix_FN.to(device).float().clone().requires_grad_(True)
        for layer, matrix_FN in init_maps_L_dict_FN.items()
    }
    if clean_cache_gb > 0:
        swap_loss.clean_cache = _clean_cache_within(
            swap_loss,
            maps_L_dict_FN,
            prompts,
            train_examples,
            [example.prompt_idx for example in [*train_examples, *heldout_examples, *monitor_examples]],
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            ids_cache=ids_cache,
            budget_gb=clean_cache_gb,
        )
    optimizer = t.optim.Adam(
        [
            {"params": [matrix_FN], "lr": relative_lr * float(matrix_FN.detach().pow(2).mean().sqrt())}
            for matrix_FN in maps_L_dict_FN.values()
        ]
    )
    scheduler = t.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))
    )
    generator = t.Generator().manual_seed(seed)

    def evaluate(example_chunks: list[list[SwapExample]]) -> tuple[float, float]:
        total_loss, hits, rows = 0.0, 0, 0
        with t.no_grad():
            for chunk in example_chunks:
                loss, hits_B = swap_loss(
                    maps_L_dict_FN,
                    SwapBatch.build(model, prompts, chunk, max_seq_len=max_seq_len, ids_cache=ids_cache),
                )
                total_loss += float(loss) * len(chunk)
                hits += int(hits_B.sum())
                rows += len(chunk)
        return total_loss / rows, hits / rows

    def heldout() -> tuple[float, float]:
        return evaluate(heldout_chunks)

    def monitor() -> dict[str, float]:
        if not monitor_chunks:
            return {}
        monitor_loss, monitor_hit_rate = evaluate(monitor_chunks)
        logger.info("    monitor: loss %.4f hit %.3f", monitor_loss, monitor_hit_rate)
        return {"monitor_loss": monitor_loss, "monitor_hit_rate": monitor_hit_rate}

    init_loss, init_hit_rate = heldout()
    best_loss, best_hit_rate, best_step = init_loss, init_hit_rate, 0
    best_L_dict_FN = {layer: matrix_FN.detach().clone() for layer, matrix_FN in maps_L_dict_FN.items()}
    logger.info("  step 0: held-out loss %.4f, hit rate %.3f", init_loss, init_hit_rate)
    history: list[dict[str, float]] = [
        {"step": 0, "heldout_loss": init_loss, "heldout_hit_rate": init_hit_rate, **monitor()}
    ]
    if log_fn is not None:
        log_fn(dict(history[0]))
    evals_without_improvement, running_loss, running_hits, start_time = 0, 0.0, 0.0, time.perf_counter()
    sorted_layers = sorted(maps_L_dict_FN)
    usable_window_sizes = [size for size in window_sizes if 1 <= size <= len(sorted_layers)]
    if all_layers_probability < 1.0 and not usable_window_sizes:
        raise ValueError("all_layers_probability < 1 needs window_sizes between 1 and the number of layers")

    def draw_active_layers() -> list[int]:
        if float(t.rand(1, generator=generator)) < all_layers_probability:
            return sorted_layers
        size = usable_window_sizes[int(t.randint(len(usable_window_sizes), (1,), generator=generator))]
        start = int(t.randint(len(sorted_layers) - size + 1, (1,), generator=generator))
        return sorted_layers[start : start + size]

    pending_batches: list[list[SwapExample]] = []

    def next_training_rows() -> list[SwapExample]:
        if not pending_batches:
            rows_Int = t.randint(len(train_examples), (batch_size * length_pool,), generator=generator).tolist()
            pool = [train_examples[i] for i in rows_Int]
            if length_pool == 1:
                return pool
            pool.sort(key=prompt_length)
            batches = [pool[start : start + batch_size] for start in range(0, len(pool), batch_size)]
            pending_batches.extend(batches[i] for i in t.randperm(len(batches), generator=generator).tolist())
        return pending_batches.pop()

    for step in range(1, steps + 1):
        batch = SwapBatch.build(model, prompts, next_training_rows(), max_seq_len=max_seq_len, ids_cache=ids_cache)
        active_layers = draw_active_layers()
        loss, hits_B = swap_loss(maps_L_dict_FN, batch, active_layers=active_layers)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += float(loss.detach())
        running_hits += float(hits_B.float().mean())
        if log_fn is not None and step % log_every == 0:
            log_fn(
                {
                    "step": step,
                    "step_train_loss": float(loss.detach()),
                    "step_train_hit_rate": float(hits_B.float().mean()),
                    "swapped_layers": len(active_layers),
                    "relative_lr": relative_lr * scheduler.get_last_lr()[0] / optimizer.param_groups[0]["initial_lr"],
                }
            )
        if step % eval_every == 0 or step == steps:
            heldout_loss, heldout_hit_rate = heldout()
            history.append(
                {
                    "step": step,
                    "train_loss": running_loss / eval_every,
                    "train_hit_rate": running_hits / eval_every,
                    "heldout_loss": heldout_loss,
                    "heldout_hit_rate": heldout_hit_rate,
                    **monitor(),
                }
            )
            logger.info(
                "  step %d: train loss %.4f hit %.3f, held-out loss %.4f hit %.3f  %.0fs",
                step,
                running_loss / eval_every,
                running_hits / eval_every,
                heldout_loss,
                heldout_hit_rate,
                time.perf_counter() - start_time,
            )
            if t.cuda.is_available():
                history[-1]["peak_gpu_memory_gb"] = t.cuda.max_memory_allocated() / 1e9
                if step == eval_every:
                    logger.info("    peak GPU memory %.1f GB", history[-1]["peak_gpu_memory_gb"])
            if log_fn is not None:
                log_fn(dict(history[-1]))
            running_loss, running_hits = 0.0, 0.0
            if heldout_loss < best_loss:
                best_loss, best_hit_rate, best_step = heldout_loss, heldout_hit_rate, step
                best_L_dict_FN = {layer: matrix_FN.detach().clone() for layer, matrix_FN in maps_L_dict_FN.items()}
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience:
                    break
    return SwapFit(
        maps_L_dict_FN={layer: matrix_FN.cpu() for layer, matrix_FN in best_L_dict_FN.items()},
        best_step=best_step,
        init_heldout_loss=init_loss,
        best_heldout_loss=best_loss,
        init_heldout_hit_rate=init_hit_rate,
        best_heldout_hit_rate=best_hit_rate,
        history=history,
    )


def _clean_cache_within(
    swap_loss: SwapLoss,
    maps_L_dict_FN: dict[int, t.Tensor],
    prompts: Sequence[str],
    train_examples: Sequence[SwapExample],
    prompt_indices: Sequence[int],
    *,
    batch_size: int,
    max_seq_len: int,
    ids_cache: dict[int, t.Tensor],
    budget_gb: float,
) -> CleanResidualCache | None:
    """A :class:`CleanResidualCache` of the prompts if it takes at most ``budget_gb`` and fits
    on the GPU next to training, else None (logged). Training's memory is measured with one
    step on the ``batch_size`` longest training rows at every layer (the costliest step); a
    tenth of the GPU is kept spare."""
    model = swap_loss.model
    length = lambda idx: len(prompt_ids(model, prompts, idx, max_seq_len=max_seq_len, cache=ids_cache))  # noqa: E731
    unique_indices = sorted(set(prompt_indices))
    num_tokens = sum(length(idx) for idx in unique_indices)
    size_gb = CleanResidualCache.size_in_bytes(model, num_tokens, len(swap_loss.layers)) / 1e9
    room_gb = float("inf")
    if t.cuda.is_available():
        longest = sorted(train_examples, key=lambda example: length(example.prompt_idx))[-batch_size:]
        t.cuda.reset_peak_memory_stats()
        loss, _ = swap_loss(maps_L_dict_FN, SwapBatch.build(model, prompts, longest, max_seq_len=max_seq_len, ids_cache=ids_cache))
        loss.backward()
        for matrix_FN in maps_L_dict_FN.values():
            matrix_FN.grad = None
        del loss
        total_gb = t.cuda.mem_get_info()[1] / 1e9
        room_gb = 0.9 * total_gb - t.cuda.max_memory_allocated() / 1e9
    if size_gb > min(budget_gb, room_gb):
        logger.info(
            "clean cache not built: it needs %.1f GB, budget %.1f GB, room next to training %.1f GB; "
            "the clean pass runs every step",
            size_gb, budget_gb, room_gb,
        )
        return None
    start_time = time.perf_counter()
    cache = CleanResidualCache(
        model, prompts, unique_indices, layers=swap_loss.layers, max_seq_len=max_seq_len, ids_cache=ids_cache
    )
    logger.info(
        "clean cache: %d prompts, %d tokens, %.1f GB (room %.1f GB), built in %.0fs",
        len(unique_indices), num_tokens, size_gb, room_gb, time.perf_counter() - start_time,
    )
    return cache


def swap_lens_config(
    *, hf_model_name: str, checkpoint_name: str, layers: Sequence[int], d_model: int,
    num_prompts_trained_on: int, max_seq_len: int, skip_first_n_positions: int,
) -> LensConfig:
    """A plain Jacobian-format lens config for swap-trained maps (read out at the final
    block, like every lens here)."""
    return LensConfig(
        hf_model_name=hf_model_name,
        checkpoint_name=checkpoint_name,
        lens_type="jacobian",
        checkpoint_every_n_prompts=None,
        source_layers=list(layers),
        d_model=d_model,
        num_prompts_trained_on=num_prompts_trained_on,
        max_seq_len=max_seq_len,
        skip_first_n_positions=skip_first_n_positions,
    )


def swap_lens(fit: SwapFit, config: LensConfig) -> JacobianLens:
    return JacobianLens(jacobians=fit.maps_L_dict_FN, config=config)


def layer_windows(layers: Sequence[int], sizes: Sequence[int]) -> dict[str, list[int]]:
    """``{name: layers}``: every single layer, every window of consecutive layers of each
    size in ``sizes`` (consecutive in the sorted list), and all of them."""
    sorted_layers = sorted(layers)
    windows: dict[str, list[int]] = {}
    for size in sizes:
        for start in range(len(sorted_layers) - size + 1):
            window = sorted_layers[start : start + size]
            name = f"L{window[0]}" if size == 1 else f"L{window[0]}-{window[-1]}"
            windows[name] = window
    windows["all"] = sorted_layers
    return windows


@t.no_grad()
def swap_hits_by_layer_set(
    model: LensModel,
    prompts: Sequence[str],
    examples: Sequence[SwapExample],
    maps_L_dict_FN: dict[int, t.Tensor],
    layer_sets: dict[str, Sequence[int]],
    *,
    excluded_mask_Bool_V: t.Tensor | None = None,
    scale: float = 1.0,
    max_seq_len: int = 128,
    batch_size: int = 128,
) -> dict[str, tuple[float, t.Tensor]]:
    """Supervised swap loss and per-example top-1 hits of ``examples`` with the swap made at
    each layer set (the probe-swap edit restricted to those layers): ``{name: (loss,
    hits_Bool_N)}``, the hits in the order of ``examples``. An empty layer set makes no edit:
    its hits are the examples whose swapped-in answer the model already gives. Works for any
    per-layer maps (a fitted lens's Jacobians, or identities for the Logit lens). The clean
    pass is run once for all layer sets (:class:`CleanResidualCache`) and the rows are batched
    by prompt length."""
    device = model_device_of(model)
    maps_on_device = {layer: matrix_FN.to(device).float() for layer, matrix_FN in maps_L_dict_FN.items()}
    ids_cache: dict[int, t.Tensor] = {}
    clean_cache = CleanResidualCache(
        model, prompts, [example.prompt_idx for example in examples], layers=list(maps_on_device),
        max_seq_len=max_seq_len, ids_cache=ids_cache,
    )
    swap_loss = SwapLoss(
        model, layers=list(maps_on_device), mode="supervised", scale=scale, window=1,
        excluded_mask_Bool_V=excluded_mask_Bool_V, clean_cache=clean_cache,
    )
    order = sorted(range(len(examples)), key=lambda idx: len(ids_cache[examples[idx].prompt_idx]))
    batches = [
        SwapBatch.build(
            model, prompts, [examples[idx] for idx in order[start : start + batch_size]],
            max_seq_len=max_seq_len, ids_cache=ids_cache,
        )
        for start in range(0, len(order), batch_size)
    ]
    order_Int_N = t.tensor(order, dtype=t.long)
    results: dict[str, tuple[float, t.Tensor]] = {}
    for name, layers in layer_sets.items():
        total_loss, hit_chunks = 0.0, []
        for batch in batches:
            loss, hits_B = swap_loss(maps_on_device, batch, active_layers=layers)
            total_loss += float(loss) * len(hits_B)
            hit_chunks.append(hits_B.cpu())
        hits_Bool_N = t.zeros(len(examples), dtype=t.bool)
        hits_Bool_N[order_Int_N] = t.cat(hit_chunks)
        results[name] = (total_loss / len(examples), hits_Bool_N)
    return results


@t.no_grad()
def swap_success_by_layer_set(
    model: LensModel,
    prompts: Sequence[str],
    examples: Sequence[SwapExample],
    maps_L_dict_FN: dict[int, t.Tensor],
    layer_sets: dict[str, Sequence[int]],
    *,
    excluded_mask_Bool_V: t.Tensor | None = None,
    scale: float = 1.0,
    max_seq_len: int = 128,
    batch_size: int = 128,
) -> dict[str, tuple[float, float]]:
    """:func:`swap_hits_by_layer_set` summarised: ``{name: (loss, hit rate)}``."""
    return {
        name: (loss, float(hits_Bool_N.float().mean()))
        for name, (loss, hits_Bool_N) in swap_hits_by_layer_set(
            model, prompts, examples, maps_L_dict_FN, layer_sets,
            excluded_mask_Bool_V=excluded_mask_Bool_V, scale=scale, max_seq_len=max_seq_len,
            batch_size=batch_size,
        ).items()
    }
