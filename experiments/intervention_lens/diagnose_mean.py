"""Do the swap-trained lenses fail because they never see the average residual?

A swap edit is ``h' - h``, so the part of ``h`` shared by every position (the clean pool's
mean ``mu``) never appears in training. Per lens and layer this prints ``|A mu|`` against
the typical ``|A (h - mu)|`` over pool positions (a large ratio means the readout is
dominated by the same vector at every position), then scores each lens with and without
subtracting ``mu`` before the map (``unembed(A (h - mu))``) on the held-out items.
Runs in the jpp_lens venv from the repo root.
"""

from __future__ import annotations

import sys

import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.intervention_fitting import CleanPool  # noqa: E402

OUT = "artifacts/gemma4_e2b"
I = f"{OUT}/intervention"
MODEL = "google/gemma-4-E2B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
CSV = f"{OUT}/model_correctness_stripped.csv"
LENS_FILES = [
    f"{I}/ilens_swap_a1.pt",
    f"{I}/ilens_swap_a0.5.pt",
    f"{I}/ilens_swap_a1_init_rlens.pt",
    f"{I}/ilens_gauss.pt",
    f"{I}/plain_jlens64.pt",
    f"{OUT}/pooled_lens.pt",
]


class CenteredLens:
    """Duck-typed lens (what ReadoutEvalRunner uses): ``A (h - mu)``."""

    def __init__(self, lens, means_L_dict_N: dict[int, t.Tensor]) -> None:
        self.config = lens.config
        self.source_layers = lens.source_layers
        self.lens = lens
        self.means_L_dict_N = means_L_dict_N

    def move_parameters_to_device(self, device) -> None:
        self.lens.move_parameters_to_device(device)
        self.means_L_dict_N = {layer: mean_N.to(device) for layer, mean_N in self.means_L_dict_N.items()}

    def transport(self, residual_PN: t.Tensor, layer: int) -> t.Tensor:
        mean_N = self.means_L_dict_N[layer].to(residual_PN.device, residual_PN.dtype)
        return self.lens.transport(residual_PN - mean_N, layer)


pool = CleanPool.load(f"{I}/clean_pool.pt")
means_L_dict_N = {layer: pool.sources_L_dict_RN[layer].float().mean(dim=0) for layer in LAYERS}
sample_rows = t.randperm(len(pool.prompt_idx_Int_R), generator=t.Generator().manual_seed(0))[:4000]

model = get_hf_model(MODEL, attn_implementation="sdpa")
lenses = jpp_cli.load_lenses(LENS_FILES, model, hf_model_name=MODEL, layers=LAYERS)

print("|A mu| / median |A (h - mu)|  (also |mu| / median |h - mu|)")
for layer in LAYERS:
    mean_N = means_L_dict_N[layer]
    centered_RN = pool.sources_L_dict_RN[layer][sample_rows].float() - mean_N
    ratios = [f"input={mean_N.norm() / centered_RN.norm(dim=-1).median():.2f}"]
    for name, lens in lenses.items():
        matrix_FN = lens.jacobians_L_dict_FN[layer].float()
        ratio = (matrix_FN @ mean_N).norm() / (centered_RN @ matrix_FN.T).norm(dim=-1).median()
        ratios.append(f"{name}={ratio:.2f}")
    print(f"L{layer:2d}  " + "  ".join(ratios), flush=True)

for name in list(lenses):
    lenses[f"{name}_centered"] = CenteredLens(lenses[name], means_L_dict_N)
_, items, excluded = jpp_cli.load_items_and_exclusions(
    model, eval_data_dir=jpp_cli.READOUT_EVALS_DATA_DIR, correctness_csv=CSV,
    hf_model_name=MODEL, layers=LAYERS, readout_max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN,
    items_spec="held-out:0",
)
ranks_df, pass_df = jpp_cli.evaluate(model, lenses, items, layers=LAYERS, excluded_token_ids=excluded, ks=[1, 10])
out = f"{I}/eval_held_out_centered.csv"
ranks_df.to_csv(out, index=False)
pass_df.to_csv(out.replace(".csv", "_pass_at_k.csv"), index=False)
table = pass_df[(pass_df["k"] == 10) & (pass_df["weighting"] == "item")]
print((100 * table.pivot(index="lens", columns="eval", values="pass_at_k")).round(0).to_string())
