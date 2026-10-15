"""Compare lens maps layer by layer, and score each lens on held-out intervention pairs.

Per lens and layer:

- ``corr`` / ``rel_dist``: entrywise Pearson correlation and relative Frobenius distance
  ``|A - A_ref| / |A_ref|`` to the ``--reference`` lens's map (the gauss sanity check uses
  the plain-gradient J-lens as the reference).
- ``err_<family>``: the held-out relative error ``sum |A delta - y|^2 / sum |y|^2`` on that
  family's held-out pairs (the same held-out prompts as ``fit-intervention``), for each
  ``--pairs-dir``. 1.0 means no better than predicting zero.

Writes a CSV. Runs in the jpp_lens venv from the repo root (GPU if there is one).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch as t

from workspace_lens.fitting.intervention_fitting import (
    LayerPairs,
    PairsMeta,
    heldout_prompt_mask,
    pairs_path,
)
from workspace_lens.utils import load_lens_file

DEVICE = "cuda" if t.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", nargs="+", required=True, help="Lens .pt files (named by stem).")
    parser.add_argument("--reference", required=True, help="Lens .pt the maps are compared to.")
    parser.add_argument("--pairs-dirs", nargs="*", default=[], help="Family pair directories.")
    parser.add_argument("--layers", required=True, help="Comma-separated layers.")
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="CSV path.")
    args = parser.parse_args()

    layers = [int(layer) for layer in args.layers.split(",")]
    lenses = {Path(path).stem: load_lens_file(path, hf_model_name=args.hf_model_name) for path in args.lens}
    reference = load_lens_file(args.reference, hf_model_name=args.hf_model_name)
    rows = {
        (name, layer): {"lens": name, "layer": layer} for name in lenses for layer in layers
    }
    for layer in layers:
        reference_FN = reference.jacobians_L_dict_FN[layer].float()
        for name, lens in lenses.items():
            matrix_FN = lens.jacobians_L_dict_FN[layer].float()
            rows[name, layer]["corr"] = float(
                t.corrcoef(t.stack([matrix_FN.flatten(), reference_FN.flatten()]))[0, 1]
            )
            rows[name, layer]["rel_dist"] = float((matrix_FN - reference_FN).norm() / reference_FN.norm())

    for pairs_dir in args.pairs_dirs:
        meta = PairsMeta.load(pairs_dir)
        heldout_prompts_Bool = heldout_prompt_mask(
            meta.num_prompts, args.holdout_fraction, seed=args.holdout_seed
        )
        for layer in layers:
            pairs = LayerPairs.load(pairs_path(pairs_dir, layer))
            heldout_Bool_P = heldout_prompts_Bool[pairs.prompt_idx_Int_P]
            delta_PN = pairs.delta_PN[heldout_Bool_P].to(DEVICE).float()
            y_PF = pairs.y_sum_PF[heldout_Bool_P].to(DEVICE).float()
            target_norm = float(y_PF.pow(2).sum())
            for name, lens in lenses.items():
                matrix_FN = lens.jacobians_L_dict_FN[layer].to(DEVICE).float()
                error = float((delta_PN @ matrix_FN.T - y_PF).pow(2).sum()) / target_norm
                rows[name, layer][f"err_{meta.family}"] = error

    table = pd.DataFrame(list(rows.values()))
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
