"""The swap-loss lens (``workspace_lens.fitting.swap_fitting``).

Pinned here:

- :func:`swap_edit` is the probe-swap eval's clamp swap
  (``lens_evals.causal_evals.runner._swapped_coordinate_clamp``), made differentiable.
- Examples never use formatting tokens as X or Y, and Y is never X.
- Descending on the swap loss lowers the training loss on the tiny Gemma model, in both
  modes, from the identity.
- The fit-swap stage runs end to end and writes a loadable Jacobian-format lens, from
  WikiText-style prompts and from a swap dataset (supervised mode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch as t

import jlens
from jlens.hf import HFLensModel
from lens_evals.causal_evals.runner import _swapped_coordinate_clamp
from lens_evals.eval_utils import formatting_token_ids_of
from workspace_lens.fitting.swap_fitting import (
    draw_swap_examples,
    fit_swap_maps,
    swap_edit,
)
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.tests.tiny_gemma_olmo import TINY_D_MODEL, build_tiny_gemma4
from workspace_lens.tests.tokenizers import CharTokenizer
from workspace_lens.utils import token_id_mask_Bool_V, vocab_size_of

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import jpp_cli  # noqa: E402
from jpp_cli import main  # noqa: E402

LAYERS = [0, 1]
MAX_SEQ_LEN = 32
PROMPTS = [
    "the quick brown fox jumps over the lazy dog and runs away",
    "a stitch in time saves nine but only when the thread holds",
    "every good boy deserves fudge said the music teacher again",
    "she sells sea shells by the sea shore on sunny afternoons",
    "many hands make light work when everyone pulls together",
    "an apple a day keeps the doctor away or so they always say",
]


@pytest.fixture(scope="module")
def model() -> HFLensModel:
    return jlens.from_hf(build_tiny_gemma4(), CharTokenizer())


@pytest.fixture(scope="module")
def excluded_mask_Bool_V(model: HFLensModel) -> t.Tensor:
    vocab_size = vocab_size_of(model)
    return token_id_mask_Bool_V(formatting_token_ids_of(model.tokenizer, vocab_size), vocab_size)


def test_swap_edit_is_the_probe_swap_clamp() -> None:
    generator = t.Generator().manual_seed(0)
    residual_1SN, clean_1SN = t.randn(2, 1, 6, 16, generator=generator)
    source_N, target_N = t.randn(2, 16, generator=generator)
    positions = [1, 2, 3, 4, 5]
    for scale in (1.0, 2.0):
        edited_1SN = swap_edit(
            residual_1SN,
            clean_1SN,
            source_N[None],
            target_N[None],
            scale=scale,
            edit_mask_Bool_S=t.arange(6) > 0,
        )
        atoms_N2 = t.stack([source_N / source_N.norm(), target_N / target_N.norm()], dim=1)
        clamp = _swapped_coordinate_clamp(atoms_N2, clean_1SN[0, positions], scale=scale, layer=0, positions=positions)
        t.testing.assert_close(edited_1SN, clamp.edit(residual_1SN, 0), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", ["next_token", "counterfactual"])
def test_examples_avoid_formatting_tokens(
    model: HFLensModel, excluded_mask_Bool_V: t.Tensor, mode: str
) -> None:
    examples = draw_swap_examples(
        model,
        PROMPTS,
        range(len(PROMPTS)),
        mode=mode,  # type: ignore[arg-type]
        examples_per_prompt=6,
        top_k=5,
        excluded_mask_Bool_V=excluded_mask_Bool_V,
        max_seq_len=MAX_SEQ_LEN,
        skip_first_n_positions=4,
        window=4,
    )
    assert examples
    for example in examples:
        assert example.source_id != example.target_id
        assert not excluded_mask_Bool_V[example.source_id] and not excluded_mask_Bool_V[example.target_id]
        assert 4 <= example.position <= MAX_SEQ_LEN - (4 if mode == "counterfactual" else 2)


@pytest.mark.parametrize("mode", ["next_token", "counterfactual"])
def test_descending_on_the_swap_loss_lowers_it(
    model: HFLensModel, excluded_mask_Bool_V: t.Tensor, mode: str
) -> None:
    draw_kwargs = dict(
        mode=mode,
        examples_per_prompt=12,
        top_k=5,
        excluded_mask_Bool_V=excluded_mask_Bool_V,
        max_seq_len=MAX_SEQ_LEN,
        skip_first_n_positions=4,
        window=4,
    )
    train_examples = draw_swap_examples(model, PROMPTS, range(4), **draw_kwargs)  # type: ignore[arg-type]
    heldout_examples = draw_swap_examples(model, PROMPTS, range(4, 6), seed=1, **draw_kwargs)  # type: ignore[arg-type]
    fit = fit_swap_maps(
        model,
        PROMPTS,
        train_examples,
        heldout_examples,
        init_maps_L_dict_FN={layer: t.eye(TINY_D_MODEL) for layer in LAYERS},
        mode=mode,  # type: ignore[arg-type]
        window=4,
        max_seq_len=MAX_SEQ_LEN,
        steps=60,
        batch_size=8,
        relative_lr=0.05,
        eval_every=20,
        patience=10,
    )
    # The training loss falls (17 training rows on a random model: the held-out loss is noise).
    train_losses = [entry["train_loss"] for entry in fit.history if "train_loss" in entry]
    assert train_losses[-1] < train_losses[0]
    assert sorted(fit.maps_L_dict_FN) == LAYERS


def test_fit_swap_stage_end_to_end(
    model: HFLensModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: model)
    monkeypatch.setattr(jpp_cli, "load_fit_prompts", lambda num_prompts: PROMPTS[:num_prompts])
    out = tmp_path / "swap_lens.pt"
    argv = ["fit-swap", "--hf-model-name", "tiny", "--layers", "0,1", "--num-prompts", "6"]
    argv += ["--mode", "counterfactual", "--max-seq-len", str(MAX_SEQ_LEN), "--skip-first-n-positions", "4"]
    argv += ["--window", "4", "--top-k", "5", "--steps", "10", "--batch-size", "4", "--eval-every", "5"]
    argv += ["--holdout-fraction", "0.34", "--checkpoint-name", "swap", "--out", str(out)]
    assert main(argv) == 0
    lens = BaseLens.load(str(out))
    assert isinstance(lens, JacobianLens) and lens.source_layers == LAYERS
    assert (tmp_path / "swap_lens_fit_history.json").exists()


SWAP_ITEMS = [
    {"name": f"{x}->{y}", "prompt": f"letter {x} goes with", "intermediate": x, "swap_to": y, "answer": a, "swap_answer": b}
    for x, a in [("q", "u"), ("j", "k"), ("v", "w"), ("f", "g")]
    for y, b in [("q", "u"), ("j", "k"), ("v", "w"), ("f", "g")]
    if x != y
] + [
    {"name": f"{x}->{y}2", "prompt": f"the {x} sound is", "intermediate": x, "swap_to": y, "answer": a, "swap_answer": b}
    for x, a in [("m", "n"), ("b", "p"), ("d", "t")]
    for y, b in [("m", "n"), ("b", "p"), ("d", "t")]
    if x != y
]


def test_fit_swap_supervised_mode_lowers_the_loss(
    model: HFLensModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows from a swap dataset (two templates, so two prompt lengths): the stage trains on
    them and the held-out loss goes down."""
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: model)
    data = tmp_path / "swap.json"
    data.write_text(__import__("json").dumps({"items": SWAP_ITEMS}))
    out = tmp_path / "supervised.pt"
    argv = ["fit-swap", "--hf-model-name", "tiny", "--layers", "0,1", "--mode", "supervised"]
    argv += ["--swap-data", str(data), "--steps", "60", "--batch-size", "4", "--eval-every", "20"]
    argv += ["--relative-lr", "0.05", "--patience", "10", "--holdout-fraction", "0.3"]
    argv += ["--monitor-data", str(data), "--checkpoint-name", "supervised", "--out", str(out)]
    assert main(argv) == 0
    stamp = __import__("json").loads((tmp_path / "supervised_stamp.json").read_text())
    assert stamp["heldout_loss"]["best"] < stamp["heldout_loss"]["init"]
    history = __import__("json").loads((tmp_path / "supervised_fit_history.json").read_text())
    assert all("monitor_loss" in entry and "monitor_hit_rate" in entry for entry in history)
    assert isinstance(BaseLens.load(str(out)), JacobianLens)


def test_fit_swap_supervised_mode_without_a_monitor(
    model: HFLensModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: model)
    data = tmp_path / "swap.json"
    data.write_text(__import__("json").dumps({"items": SWAP_ITEMS}))
    argv = ["fit-swap", "--hf-model-name", "tiny", "--layers", "0,1", "--mode", "supervised"]
    argv += ["--swap-data", str(data), "--steps", "4", "--batch-size", "4", "--eval-every", "2"]
    argv += ["--holdout-fraction", "0.3", "--checkpoint-name", "s", "--out", str(tmp_path / "s.pt")]
    assert main(argv) == 0


def test_layer_windows_cover_singles_windows_and_all() -> None:
    from workspace_lens.fitting.swap_fitting import layer_windows

    windows = layer_windows([8, 4, 12, 16], [1, 3])
    assert windows == {
        "L4": [4], "L8": [8], "L12": [12], "L16": [16],
        "L4-12": [4, 8, 12], "L8-16": [8, 12, 16],
        "all": [4, 8, 12, 16],
    }


def test_window_training_and_per_layer_set_eval(model: HFLensModel, excluded_mask_Bool_V: t.Tensor) -> None:
    """Training that swaps at random single layers some of the time runs, and the
    per-layer-set evaluation scores every set."""
    from workspace_lens.fitting.swap_fitting import layer_windows, swap_success_by_layer_set

    prompts, examples = jpp_cli.swap_examples_from_items(model, SWAP_ITEMS, max_seq_len=MAX_SEQ_LEN)
    fit = fit_swap_maps(
        model,
        prompts,
        examples[:12],
        examples[12:],
        init_maps_L_dict_FN={layer: t.eye(TINY_D_MODEL) for layer in LAYERS},
        mode="supervised",
        max_seq_len=MAX_SEQ_LEN,
        steps=10,
        batch_size=4,
        eval_every=5,
        excluded_mask_Bool_V=excluded_mask_Bool_V,
        window_sizes=[1],
        all_layers_probability=0.5,
    )
    windows = layer_windows(LAYERS, [1])
    results = swap_success_by_layer_set(
        model, prompts, examples[12:], fit.maps_L_dict_FN, windows,
        excluded_mask_Bool_V=excluded_mask_Bool_V, max_seq_len=MAX_SEQ_LEN,
    )
    assert set(results) == {"L0", "L1", "all"}
    assert all(0.0 <= hit_rate <= 1.0 for _, hit_rate in results.values())


def test_padded_mixed_length_batch_matches_separate_batches(
    model: HFLensModel, excluded_mask_Bool_V: t.Tensor
) -> None:
    """Right-padding rows of different lengths changes nothing a row reads: the loss of a
    mixed batch is the mean of the per-row losses taken unpadded."""
    from workspace_lens.fitting.swap_fitting import SwapBatch, SwapLoss

    prompts, examples = jpp_cli.swap_examples_from_items(model, SWAP_ITEMS, max_seq_len=MAX_SEQ_LEN)
    lengths = {len(prompts[example.prompt_idx]) for example in examples}
    assert len(lengths) > 1
    maps = {layer: t.eye(TINY_D_MODEL) + 0.1 * t.randn(TINY_D_MODEL, TINY_D_MODEL, generator=t.Generator().manual_seed(layer)) for layer in LAYERS}
    swap_loss = SwapLoss(model, layers=LAYERS, mode="supervised", scale=1.0, window=1, excluded_mask_Bool_V=excluded_mask_Bool_V)
    with t.no_grad():
        mixed_loss, mixed_hits = swap_loss(maps, SwapBatch.build(model, prompts, examples, max_seq_len=MAX_SEQ_LEN))
        single = [swap_loss(maps, SwapBatch.build(model, prompts, [example], max_seq_len=MAX_SEQ_LEN)) for example in examples]
    t.testing.assert_close(mixed_loss, t.stack([loss for loss, _ in single]).mean(), atol=1e-4, rtol=1e-4)
    assert mixed_hits.tolist() == [bool(hits[0]) for _, hits in single]


def test_log_fn_gets_step_and_evaluation_metrics(model: HFLensModel, excluded_mask_Bool_V: t.Tensor) -> None:
    prompts, examples = jpp_cli.swap_examples_from_items(model, SWAP_ITEMS, max_seq_len=MAX_SEQ_LEN)
    logged: list[dict] = []
    fit_swap_maps(
        model, prompts, examples[:12], examples[12:],
        init_maps_L_dict_FN={layer: t.eye(TINY_D_MODEL) for layer in LAYERS},
        mode="supervised", max_seq_len=MAX_SEQ_LEN, steps=6, batch_size=4, eval_every=3,
        excluded_mask_Bool_V=excluded_mask_Bool_V, log_fn=logged.append, log_every=2,
    )
    steps_with_train_loss = [entry["step"] for entry in logged if "step_train_loss" in entry]
    steps_with_heldout = [entry["step"] for entry in logged if "heldout_loss" in entry]
    assert steps_with_train_loss == [2, 4, 6] and steps_with_heldout == [0, 3, 6]


def test_length_pool_cuts_padding(
    model: HFLensModel, excluded_mask_Bool_V: t.Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grouping the rows of several batches by prompt length (two lengths here) leaves less
    padding in the batches than drawing each batch on its own."""
    from workspace_lens.fitting import swap_fitting

    prompts, examples = jpp_cli.swap_examples_from_items(model, SWAP_ITEMS, max_seq_len=MAX_SEQ_LEN)
    build = swap_fitting.SwapBatch.build

    def padding_with(length_pool: int) -> int:
        padded: list[int] = []

        def recording_build(*args, **kwargs):
            batch = build(*args, **kwargs)
            padded.append(batch.input_ids_Int_BS.numel() - int((batch.position_Int_B + 1).sum()))
            return batch

        monkeypatch.setattr(swap_fitting.SwapBatch, "build", recording_build)
        fit_swap_maps(
            model, prompts, examples[:16], examples[16:],
            init_maps_L_dict_FN={layer: t.eye(TINY_D_MODEL) for layer in LAYERS},
            mode="supervised", max_seq_len=MAX_SEQ_LEN, steps=16, batch_size=4, eval_every=16,
            excluded_mask_Bool_V=excluded_mask_Bool_V, length_pool=length_pool,
        )
        return sum(padded)

    assert padding_with(8) < padding_with(1)


def test_clean_cache_matches_the_clean_pass(model: HFLensModel, excluded_mask_Bool_V: t.Tensor) -> None:
    """The stored clean residuals are the clean pass's at every real position of a mixed-length
    batch, and the swap loss reading them equals the loss that runs the clean pass."""
    from workspace_lens.fitting.swap_fitting import CleanResidualCache, SwapBatch, SwapLoss

    prompts, examples = jpp_cli.swap_examples_from_items(model, SWAP_ITEMS, max_seq_len=MAX_SEQ_LEN)
    ids_cache: dict[int, t.Tensor] = {}
    cache = CleanResidualCache(
        model, prompts, [example.prompt_idx for example in examples], layers=LAYERS,
        max_seq_len=MAX_SEQ_LEN, ids_cache=ids_cache, batch_size=3,
    )
    batch = SwapBatch.build(model, prompts, examples, max_seq_len=MAX_SEQ_LEN, ids_cache=ids_cache)
    maps = {layer: t.eye(TINY_D_MODEL) + 0.1 * t.randn(TINY_D_MODEL, TINY_D_MODEL, generator=t.Generator().manual_seed(layer)) for layer in LAYERS}
    swap_loss = SwapLoss(model, layers=LAYERS, mode="supervised", scale=1.0, window=1, excluded_mask_Bool_V=excluded_mask_Bool_V)
    with t.no_grad():
        clean_L_dict_BSN, _ = swap_loss._targets(batch, LAYERS)
        cached_L_dict_BSN = cache.residuals(batch, LAYERS)
        real_Bool_BS = t.arange(batch.input_ids_Int_BS.shape[1]) <= batch.position_Int_B[:, None]
        for layer in LAYERS:
            t.testing.assert_close(cached_L_dict_BSN[layer][real_Bool_BS], clean_L_dict_BSN[layer][real_Bool_BS])
        loss, hits = swap_loss(maps, batch)
        swap_loss.clean_cache = cache
        cached_loss, cached_hits = swap_loss(maps, batch)
    t.testing.assert_close(cached_loss, loss)
    assert cached_hits.tolist() == hits.tolist()


def test_swap_edit_in_bf16_is_close_to_fp32() -> None:
    from workspace_lens.fitting.swap_fitting import swap_edit

    generator = t.Generator().manual_seed(0)
    residual_BSN, clean_BSN = t.randn(2, 4, 6, 32, generator=generator)
    source_BN, target_BN = t.randn(2, 4, 32, generator=generator)
    mask_Bool_S = t.arange(6) > 0
    edited_fp32 = swap_edit(residual_BSN, clean_BSN, source_BN, target_BN, scale=1.0, edit_mask_Bool_S=mask_Bool_S)
    edited_bf16 = swap_edit(
        residual_BSN.bfloat16(), clean_BSN.bfloat16(), source_BN, target_BN, scale=1.0, edit_mask_Bool_S=mask_Bool_S
    )
    assert edited_bf16.dtype == t.bfloat16
    t.testing.assert_close(edited_bf16.float(), edited_fp32, atol=0.05, rtol=0.02)
