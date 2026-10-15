"""The intervention-trained lens (``workspace_lens.fitting.intervention_fitting``) on
the tiny CPU Gemma 4 model (real attention, so an edit reaches later positions).

Pinned here:

- Edit families parse from their CLI spelling and name their pair directories.
- Each pair's summed target matches the J-lens estimator's first-order prediction for
  a small Gaussian edit (the hook edits the right row, block and position, and the
  right positions are summed), and nothing changes before the edited position.
- A swap edit puts the donor residual in place exactly (alpha 1).
- :func:`fit_transport` recovers a planted linear map.
- The ``gauss`` family fitted from the identity recovers the plain J-lens
  (:class:`LensTrainer`) on the same prompts: the spec's built-in check.
- The CLI stages run end to end, the fitted lens loads as a ``JacobianLens``, and
  ``build-experts`` stacks whole lenses as experts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch as t

import jlens
from jlens.hf import HFLensModel
from workspace_lens.fitting.condense_experts import ExpertJacobians
from workspace_lens.fitting.intervention_fitting import (
    CleanPool,
    EditFamily,
    LayerPairs,
    collect_pairs,
    fit_transport,
    projected_jacobian_errors,
    record_clean_pool,
)
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.tests.fixtures import TINY_D_MODEL as TINY_D_MODEL_8
from workspace_lens.tests.fixtures import (
    TINY_FIT_PROMPTS,
    TINY_SOURCE_LAYERS,
    make_tiny_decoder,
    make_tiny_lens_config,
)
from workspace_lens.tests.tiny_gemma_olmo import build_tiny_gemma4
from workspace_lens.tests.tokenizers import CharTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import jpp_cli  # noqa: E402
from jpp_cli import main  # noqa: E402

LAYERS = [0, 1]
TARGET_LAYER = 2
MAX_SEQ_LEN = 48
SKIP_FIRST_N_POSITIONS = 4
PROMPTS = [
    "the quick brown fox jumps over the lazy dog and runs away",
    "a stitch in time saves nine but only when the thread holds",
    "every good boy deserves fudge said the music teacher again",
    "she sells sea shells by the sea shore on sunny afternoons",
]


@pytest.fixture(scope="module")
def model() -> HFLensModel:
    return jlens.from_hf(build_tiny_gemma4(), CharTokenizer())


@pytest.fixture(scope="module")
def pool(model: HFLensModel) -> CleanPool:
    return record_clean_pool(
        model,
        PROMPTS,
        layers=LAYERS,
        target_layer=TARGET_LAYER,
        max_seq_len=MAX_SEQ_LEN,
        skip_first_n_positions=SKIP_FIRST_N_POSITIONS,
    )


def _collect(model: HFLensModel, pool: CleanPool, families: list[EditFamily], *, pairs_per_prompt: int):
    return collect_pairs(
        model,
        PROMPTS,
        pool,
        families=families,
        layers=LAYERS,
        target_layer=TARGET_LAYER,
        pairs_per_prompt=pairs_per_prompt,
        rows_per_pass=32,
        max_seq_len=MAX_SEQ_LEN,
        skip_first_n_positions=SKIP_FIRST_N_POSITIONS,
    )


def test_edit_families_parse_and_label() -> None:
    assert EditFamily.parse("swap:0.5") == EditFamily("swap", alpha=0.5)
    assert EditFamily.parse("swap").label == "swap_a1"
    assert EditFamily.parse("gauss:0.01").label == "gauss_s0.01"
    assert EditFamily.parse("zero").label == "zero"
    with pytest.raises(ValueError):
        EditFamily.parse("blur:2")


def test_small_gaussian_pairs_match_the_jlens_first_order_prediction(
    model: HFLensModel, pool: CleanPool
) -> None:
    pairs_by_family, stats = _collect(model, pool, [EditFamily("gauss", size=1e-3)], pairs_per_prompt=6)
    assert stats.max_leak == 0.0
    for layer in LAYERS:
        layer_pairs = pairs_by_family["gauss_s0.001"][layer]
        from_first_prompt = layer_pairs.prompt_idx_Int_P == 0
        first_prompt_pairs = LayerPairs(
            *(getattr(layer_pairs, name)[from_first_prompt] for name in LayerPairs.__dataclass_fields__)
        )
        errors_P = projected_jacobian_errors(
            model,
            PROMPTS[0],
            first_prompt_pairs,
            layer=layer,
            target_layer=TARGET_LAYER,
            max_seq_len=MAX_SEQ_LEN,
            skip_first_n_positions=SKIP_FIRST_N_POSITIONS,
        )
        # The error is the second-order term (it shrinks tenfold from size 1e-2 to 1e-3 on
        # this model); a wrong position or summing range would make it O(1).
        assert errors_P.max() < 3e-2, errors_P


def test_full_swap_writes_the_donor_residual(model: HFLensModel, pool: CleanPool) -> None:
    """``h + delta`` of an alpha-1 swap is a residual of another prompt from the pool."""
    pairs_by_family, _ = _collect(model, pool, [EditFamily("swap")], pairs_per_prompt=3)
    layer = LAYERS[1]
    layer_pairs = pairs_by_family["swap_a1"][layer]
    sources_RN = pool.sources_L_dict_RN[layer].float()
    for pair_idx in range(len(layer_pairs.delta_PN)):
        prompt_idx = int(layer_pairs.prompt_idx_Int_P[pair_idx])
        position = int(layer_pairs.position_Int_P[pair_idx])
        clean_row = int(t.where((pool.prompt_idx_Int_R == prompt_idx) & (pool.position_Int_R == position))[0])
        edited_N = sources_RN[clean_row] + layer_pairs.delta_PN[pair_idx].float()
        distances_R = (sources_RN - edited_N).norm(dim=-1) / sources_RN[clean_row].norm()
        nearest_row = int(distances_R.argmin())
        assert distances_R[nearest_row] < 1e-2
        assert int(pool.prompt_idx_Int_R[nearest_row]) != prompt_idx


def test_fit_transport_recovers_a_planted_map() -> None:
    generator = t.Generator().manual_seed(0)
    planted_FN = t.eye(16) + 0.3 * t.randn(16, 16, generator=generator)
    delta_PN = t.randn(4000, 16, generator=generator) * t.linspace(0.2, 3.0, 16)
    y_PF = delta_PN @ planted_FN.T
    fit = fit_transport(
        delta_PN,
        y_PF,
        t.arange(4000) % 10 == 0,
        init_FN=t.eye(16),
        steps=3000,
        lr=1e-2,
        batch_size=256,
    )
    assert fit.best_heldout_relative_error < 1e-4
    assert (fit.matrix_FN - planted_FN).norm() / planted_FN.norm() < 1e-2


def test_gauss_family_recovers_the_plain_jlens(tmp_path: Path) -> None:
    """The spec's built-in check, end to end: Gaussian edits, residual loss, identity init
    -> the J-lens fitted by backprop. Run on TinyDecoder, whose linear blocks make every
    position's Jacobian the same, so the regression has no sampling noise and the match
    must be close to exact. (On a model whose Jacobian varies by position the gauss fit is a
    noisy estimate of the mean; its first-order exactness per pair is pinned above.)"""
    decoder = make_tiny_decoder()
    config = make_tiny_lens_config(tmp_path, "plain")
    tiny_pool = record_clean_pool(
        decoder,
        TINY_FIT_PROMPTS,
        layers=TINY_SOURCE_LAYERS,
        target_layer=decoder.n_layers - 1,
        max_seq_len=config.max_seq_len,
        skip_first_n_positions=config.skip_first_n_positions,
    )
    pairs_by_family, _ = collect_pairs(
        decoder,
        TINY_FIT_PROMPTS,
        tiny_pool,
        families=[EditFamily("gauss")],
        layers=TINY_SOURCE_LAYERS,
        target_layer=decoder.n_layers - 1,
        pairs_per_prompt=64,
        rows_per_pass=64,
        max_seq_len=config.max_seq_len,
        skip_first_n_positions=config.skip_first_n_positions,
    )
    plain_jlens = LensTrainer(config, decoder, TINY_FIT_PROMPTS).fit()
    for layer in TINY_SOURCE_LAYERS:
        layer_pairs = pairs_by_family["gauss_s0.05"][layer]
        fit = fit_transport(
            layer_pairs.delta_PN,
            layer_pairs.y_sum_PF,
            layer_pairs.prompt_idx_Int_P == 1,
            init_FN=t.eye(TINY_D_MODEL_8),
            steps=3000,
            lr=3e-3,
            batch_size=64,
        )
        jacobian_FN = plain_jlens.jacobians_L_dict_FN[layer]
        assert (fit.matrix_FN - jacobian_FN).norm() / jacobian_FN.norm() < 1e-2, layer


def test_cli_stages_end_to_end(
    model: HFLensModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: model)
    monkeypatch.setattr(jpp_cli, "load_fit_prompts", lambda num_prompts: PROMPTS[:num_prompts])
    pairs_dir = tmp_path / "pairs"
    common = ["--hf-model-name", "tiny", "--layers", "0,1"]
    assert (
        main(
            [
                "collect-pairs",
                *common,
                "--num-prompts",
                "4",
                "--families",
                "gauss:0.01,swap:1,zero",
                "--pairs-per-prompt",
                "8",
                "--rows-per-pass",
                "16",
                "--max-seq-len",
                str(MAX_SEQ_LEN),
                "--skip-first-n-positions",
                str(SKIP_FIRST_N_POSITIONS),
                "--out-dir",
                str(pairs_dir),
            ]
        )
        == 0
    )
    assert (pairs_dir / "clean_pool.pt").exists() and (pairs_dir / "gram.pt").exists()
    lens_paths = []
    for family, loss in [("swap_a1", "residual"), ("gauss_s0.01", "logit")]:
        lens_path = tmp_path / f"{family}.pt"
        assert (
            main(
                [
                    "fit-intervention",
                    "--layers",
                    "0,1",
                    "--pairs-dir",
                    str(pairs_dir / family),
                    "--loss",
                    loss,
                    "--steps",
                    "40",
                    "--batch-size",
                    "16",
                    "--eval-every",
                    "10",
                    "--holdout-fraction",
                    "0.25",
                    "--device",
                    "cpu",
                    "--checkpoint-name",
                    family,
                    "--out",
                    str(lens_path),
                ]
            )
            == 0
        )
        lens = BaseLens.load(str(lens_path))
        assert isinstance(lens, JacobianLens) and lens.source_layers == LAYERS
        assert (tmp_path / f"{family}_fit_history.json").exists()
        lens_paths.append(str(lens_path))

    # Several families' pairs pooled into one fit (swap plus removal edits).
    pooled_path = tmp_path / "swap_and_zero.pt"
    fit_argv = ["fit-intervention", "--layers", "0,1", "--steps", "20", "--batch-size", "16"]
    fit_argv += ["--eval-every", "10", "--holdout-fraction", "0.25", "--device", "cpu"]
    fit_argv += ["--pairs-dir", str(pairs_dir / "swap_a1"), str(pairs_dir / "zero")]
    assert main([*fit_argv, "--checkpoint-name", "pooled", "--out", str(pooled_path)]) == 0
    assert BaseLens.load(str(pooled_path)).source_layers == LAYERS
    with pytest.raises(ValueError, match="num_prompts"):
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "meta.json").write_text(
            (pairs_dir / "zero" / "meta.json").read_text().replace('"num_prompts": 4', '"num_prompts": 5')
        )
        main([*fit_argv[:-2], str(pairs_dir / "swap_a1"), str(other_dir), "--checkpoint-name", "x", "--out", str(tmp_path / "x.pt")])

    kl_path = tmp_path / "output_kl.pt"
    assert (
        main(
            [
                "fit-output-kl",
                *common,
                "--pool-dir",
                str(pairs_dir),
                "--steps",
                "20",
                "--batch-size",
                "16",
                "--eval-every",
                "10",
                "--holdout-fraction",
                "0.25",
                "--init",
                lens_paths[0],
                "--checkpoint-name",
                "output_kl",
                "--out",
                str(kl_path),
            ]
        )
        == 0
    )
    lens_paths.append(str(kl_path))

    experts_path = tmp_path / "experts.pt"
    assert (
        main(["build-experts", *common, "--lens", *lens_paths, "--out", str(experts_path)]) == 0
    )
    experts = ExpertJacobians.load(str(experts_path))
    assert experts.num_experts == 3 and experts.layers == LAYERS
    combined = experts.combine({layer: t.tensor([0.0, 0.0, 1.0]) for layer in LAYERS}, checkpoint_name="kl")
    kl_lens = BaseLens.load(str(kl_path))
    for layer in LAYERS:
        t.testing.assert_close(
            combined.jacobians_L_dict_FN[layer],
            kl_lens.jacobians_L_dict_FN[layer].half().float(),
        )
