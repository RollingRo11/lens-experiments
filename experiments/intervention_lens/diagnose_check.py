"""Does the J-lens's first-order prediction of a small edit's effect get better as the edit
shrinks? For a smooth model the relative error of the first-order prediction falls in
proportion to the edit size (halve the edit, halve the error), until rounding takes over.

Collects Gaussian edits of several sizes (|delta| = size * |h|) on ``--num-prompts`` WikiText
prompts, with the model in bf16 and in fp32, and scores every pair of every prompt with the
J-lens check (``projected_jacobian_errors``: the relative error of ``G_s delta`` against the
measured summed change). Prints, per layer, the median error at each size, how much it fell
from the previous size next to how much the size fell, and the slope of log(error) against
log(size) (1: proportional, as a smooth model's must be; 0: flat). Writes every pair's error
to ``--out``. Runs in the jpp_lens venv from the repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

import jlens  # noqa: E402
from workspace_lens.fitting.intervention_fitting import (  # noqa: E402
    EditFamily,
    LayerPairs,
    collect_pairs,
    projected_jacobian_errors,
    record_clean_pool,
)

MODEL = "google/gemma-4-E2B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]


def pairs_of_prompt(pairs: LayerPairs, prompt_idx: int) -> LayerPairs:
    keep_Bool_P = pairs.prompt_idx_Int_P == prompt_idx
    return LayerPairs(*(getattr(pairs, name)[keep_Bool_P] for name in LayerPairs.__dataclass_fields__))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument("--sizes", default="0.05,0.01,0.002,0.0005")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--pairs-per-prompt", type=int, default=8)
    parser.add_argument("--out", required=True, help="CSV of every pair's error.")
    args = parser.parse_args()
    sizes = [float(size) for size in args.sizes.split(",")]
    prompts = jpp_cli.load_fit_prompts(args.num_prompts)
    rows = []
    for dtype_name in args.dtypes.split(","):
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=getattr(t, dtype_name), attn_implementation="sdpa"
        ).cuda()
        model = jlens.from_hf(hf_model, AutoTokenizer.from_pretrained(MODEL))
        target_layer = model.n_layers - 1
        pool = record_clean_pool(
            model, prompts, layers=LAYERS, target_layer=target_layer, max_seq_len=128, skip_first_n_positions=16
        )
        families = [EditFamily("gauss", size=size) for size in sizes]
        pairs_by_family, stats = collect_pairs(
            model,
            prompts,
            pool,
            families=families,
            layers=LAYERS,
            target_layer=target_layer,
            pairs_per_prompt=args.pairs_per_prompt,
            rows_per_pass=128,
            max_seq_len=128,
            skip_first_n_positions=16,
        )
        print(f"== {dtype_name}  max_leak={stats.max_leak:.3g}", flush=True)
        for family in families:
            for layer in LAYERS:
                pairs = pairs_by_family[family.label][layer]
                for prompt_idx in sorted(set(pairs.prompt_idx_Int_P.tolist())):
                    errors_P = projected_jacobian_errors(
                        model,
                        prompts[prompt_idx],
                        pairs_of_prompt(pairs, prompt_idx),
                        layer=layer,
                        target_layer=target_layer,
                        max_seq_len=128,
                        skip_first_n_positions=16,
                    )
                    rows += [
                        {"dtype": dtype_name, "size": family.size, "layer": layer, "prompt": prompt_idx, "error": float(e)}
                        for e in errors_P
                    ]
        del model, hf_model
        t.cuda.empty_cache()

    errors_df = pd.DataFrame(rows)
    errors_df.to_csv(args.out, index=False)
    medians = errors_df.groupby(["dtype", "layer", "size"])["error"].median()
    counts = errors_df.groupby(["dtype", "layer", "size"])["error"].size()
    print(f"\npairs per (dtype, layer, size): {counts.min()} to {counts.max()}", flush=True)
    for dtype_name in errors_df["dtype"].unique():
        print(f"\n== {dtype_name}: median relative error of the first-order prediction", flush=True)
        for layer in LAYERS:
            by_size = medians.loc[(dtype_name, layer)].sort_index(ascending=False)  # largest edit first
            cells = [f"{size:g}: {error:.4f}" for size, error in by_size.items()]
            shrink = [
                f"x{by_size.iloc[i - 1] / by_size.iloc[i]:.1f} (size x{by_size.index[i - 1] / by_size.index[i]:g})"
                for i in range(1, len(by_size))
            ]
            slope = np.polyfit(np.log(by_size.index.to_numpy()), np.log(by_size.to_numpy()), 1)[0]
            print(f"L{layer:<3d} {'  '.join(cells)}  | error fell {', '.join(shrink)} | log-log slope {slope:.2f}",
                  flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
