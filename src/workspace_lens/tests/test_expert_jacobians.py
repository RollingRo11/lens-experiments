"""CPU tests for ``fitting/condense_experts/expert_jacobians.py``:

1. ``combine_expert_jacobians`` is the explicit weighted sum, and its transport
   equals the weighted sum of the per-expert transports (linearity).
2. ``ExpertJacobians.combine`` transports as ``sum_e w_e J_e x`` (fp16-stored
   experts combine as their fp32 casts), resets the router config fields,
   rejects unknown layers, round-trips through ``BaseLens.save(dtype=float32)``
   / ``load`` bit for bit.
3. ``ExpertJacobians.collapse_fallbacks`` merges a layer's pooled copies into one
   slot, keeps layers without fallbacks as they are, and reaches the same lenses.

The fake ``ExpertJacobians`` / ``ReadoutResiduals`` (fake tensors in the real
dataclasses, ``TinyDecoder`` as the model) come from ``fixtures.py`` and are
shared with ``test_condense_fitter.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from workspace_lens.fitting.condense_experts.expert_jacobians import (
    combine_expert_jacobians,
)
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.tests.fixtures import (
    CONDENSE_D_MODEL,
    CONDENSE_LAYERS,
    CONDENSE_NUM_EXPERTS,
    CONDENSE_VOCAB_SIZE,
    all_item_expert_transports,
    make_fake_expert_jacobians,
    make_fake_readout_residuals,
)


def test_combine_expert_jacobians_is_the_weighted_sum_and_commutes_with_transport() -> None:
    generator = t.Generator().manual_seed(5)
    experts_EFN = t.randn(CONDENSE_NUM_EXPERTS, CONDENSE_D_MODEL, CONDENSE_D_MODEL, generator=generator)
    weights_E = t.tensor([0.5, -0.2, 0.0, 1.5])
    combined_FN = combine_expert_jacobians(experts_EFN, weights_E)
    explicit_FN = t.zeros(CONDENSE_D_MODEL, CONDENSE_D_MODEL)
    for expert_idx in range(CONDENSE_NUM_EXPERTS):
        explicit_FN += float(weights_E[expert_idx]) * experts_EFN[expert_idx]
    assert combined_FN.shape == (CONDENSE_D_MODEL, CONDENSE_D_MODEL)
    assert t.allclose(combined_FN, explicit_FN, atol=1e-6)
    # Linearity: (sum_e w_e J_e) x == sum_e w_e (J_e x), so the inference lens's
    # transport is the weighted prediction the weights were fitted on.
    residuals_PN = t.randn(7, CONDENSE_D_MODEL, generator=generator)
    transports_PEF = t.einsum("pj,eij->pei", residuals_PN, experts_EFN)
    assert t.allclose(
        residuals_PN @ combined_FN.T,
        t.einsum("pen,e->pn", transports_PEF, weights_E),
        atol=1e-5,
    )


### WIRING: ExpertJacobians / ReadoutResiduals / the combined lens

def test_combine_builds_the_weighted_jacobian_lens(tmp_path: Path) -> None:
    generator = t.Generator().manual_seed(10)
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    residuals = make_fake_readout_residuals(generator)
    weights_L_dict_E = {
        1: t.tensor([0.5, -0.2, 0.0, 1.5]),
        2: t.tensor([0.3, 0.3, 0.3, 0.3]),
    }
    lens = experts.combine(weights_L_dict_E, checkpoint_name="jpp_lens")
    assert isinstance(lens, JacobianLens)
    assert lens.source_layers == CONDENSE_LAYERS
    for layer in CONDENSE_LAYERS:
        residuals_PN = residuals.residuals_L_dict_PN[layer]
        expected_PF = t.einsum(
            "pen,e->pn",
            all_item_expert_transports(model, experts, residuals, layer),
            weights_L_dict_E[layer],
        )
        assert t.allclose(lens.transport(residuals_PN, layer), expected_PF, atol=1e-4)

    # Experts stored in fp16 (as ExpertJacobians.load returns them) combine as
    # their fp32 casts, in fp32.
    half_experts = dataclasses.replace(
        experts,
        experts_L_dict_EFN={
            layer: experts_EFN.half()
            for layer, experts_EFN in experts.experts_L_dict_EFN.items()
        },
    )
    half_lens = half_experts.combine(weights_L_dict_E, checkpoint_name="h")
    for layer in CONDENSE_LAYERS:
        assert half_lens.jacobians_L_dict_FN[layer].dtype == t.float32
        assert t.equal(
            half_lens.jacobians_L_dict_FN[layer],
            combine_expert_jacobians(
                experts.experts_L_dict_EFN[layer].half().float(), weights_L_dict_E[layer]
            ),
        )

    # The router fields are reset, the name replaced, everything else kept —
    # as build_matched_baseline_lens does.
    config, source_config = lens.config, experts.config
    assert config.lens_type == "jacobian"
    assert config.num_clusters == 0
    assert config.cluster_projection_dim == 0
    assert config.checkpoint_name == "jpp_lens"
    for field_name in (
        "hf_model_name",
        "source_layers",
        "relative_end_transport_layer",
        "d_model",
        "num_prompts_trained_on",
        "lrp_mode",
        "creation_time",
        "artifacts_base_dir",
    ):
        assert getattr(config, field_name) == getattr(source_config, field_name)

    # A subset of the experts' layers is fine; an unknown layer is not.
    assert experts.combine({1: weights_L_dict_E[1]}, checkpoint_name="one").source_layers == [
        1
    ]
    with pytest.raises(ValueError, match=r"layers \[5\]"):
        experts.combine(
            {1: weights_L_dict_E[1], 5: weights_L_dict_E[1]},
            checkpoint_name="x",
        )

    # fp32 save / load reproduces the combined Jacobians bit for bit.
    path = str(tmp_path / "jpp_lens.pt")
    lens.save(path, dtype=t.float32)
    loaded = BaseLens.load(path)
    assert isinstance(loaded, JacobianLens)
    assert loaded.source_layers == CONDENSE_LAYERS
    assert loaded.jacobians_L_dict_FN is not None and lens.jacobians_L_dict_FN is not None
    for layer in CONDENSE_LAYERS:
        assert t.equal(loaded.jacobians_L_dict_FN[layer], lens.jacobians_L_dict_FN[layer])
    assert loaded.config.to_dict() == lens.config.to_dict()


def test_collapse_fallbacks_merges_pooled_copies_into_one_slot(tmp_path: Path) -> None:
    """A layer with fallback experts (copies of the pooled Jacobian) keeps its own experts
    plus one pooled slot holding the fallback clusters' counts; a layer without fallbacks
    is unchanged. Any weighting of the stored slots equals the collapsed weighting whose
    pooled weight is the sum of the copies' weights."""
    generator = t.Generator().manual_seed(21)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    collapsed_layer, other_layer = CONDENSE_LAYERS[0], CONDENSE_LAYERS[1]
    fallback_Bool_E = t.tensor([False, True, False, True])
    experts.experts_L_dict_EFN[collapsed_layer][fallback_Bool_E] = experts.pooled_jacobian_L_dict_FN[
        collapsed_layer
    ]
    experts.fallback_L_dict_Bool_E[collapsed_layer] = fallback_Bool_E
    experts.position_counts_L_dict_E[collapsed_layer] = t.tensor([300, 7, 120, 0])
    experts.weight_sums_L_dict_E[collapsed_layer] = t.tensor([300.0, 7.0, 120.0, 0.0])

    summary = experts.layer_summary()[collapsed_layer]
    assert summary["populated"] == 3 and summary["own_jacobian"] == 2
    assert summary["fallback_slots"] == 2
    assert summary["fallback_position_fraction"] == pytest.approx(7 / 427)

    collapsed = experts.collapse_fallbacks()
    assert collapsed.experts_L_dict_EFN[collapsed_layer].shape[0] == 3
    assert t.equal(collapsed.experts_L_dict_EFN[collapsed_layer][:2], experts.experts_L_dict_EFN[collapsed_layer][[0, 2]])
    assert t.equal(collapsed.experts_L_dict_EFN[collapsed_layer][2], experts.pooled_jacobian_L_dict_FN[collapsed_layer])
    assert collapsed.position_counts_L_dict_E[collapsed_layer].tolist() == [300, 120, 7]
    assert collapsed.fallback_L_dict_Bool_E[collapsed_layer].tolist() == [False, False, True]
    assert collapsed.experts_L_dict_EFN[other_layer] is experts.experts_L_dict_EFN[other_layer]

    weights_E = t.tensor([0.4, -0.3, 0.2, 0.5])
    collapsed_weights_E = t.tensor([0.4, 0.2, -0.3 + 0.5])
    original_lens = experts.combine({collapsed_layer: weights_E}, checkpoint_name="original")
    collapsed_lens = collapsed.combine({collapsed_layer: collapsed_weights_E}, checkpoint_name="collapsed")
    assert t.allclose(
        original_lens.jacobians_L_dict_FN[collapsed_layer],
        collapsed_lens.jacobians_L_dict_FN[collapsed_layer],
        atol=1e-5,
    )
