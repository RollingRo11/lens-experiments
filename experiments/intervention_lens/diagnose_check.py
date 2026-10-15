"""Why do Gemma's gauss pairs miss the J-lens's first-order prediction? Separates bf16
noise from edit size: the J-lens check (``projected_jacobian_errors``) for Gaussian edits
of several sizes, with the model in bf16 and in fp32. If fp32 at small sizes matches,
the hook and positions are right and the bf16 / size numbers say what the gauss family
can recover. Runs in the jpp_lens venv from the repo root.
"""

from __future__ import annotations

import argparse
import sys

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
SIZES = [0.05, 0.01, 0.002, 0.0005]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="bfloat16,float32")
    args = parser.parse_args()
    prompts = jpp_cli.load_fit_prompts(4)
    for dtype_name in args.dtypes.split(","):
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=getattr(t, dtype_name), attn_implementation="sdpa"
        ).cuda()
        model = jlens.from_hf(hf_model, AutoTokenizer.from_pretrained(MODEL))
        target_layer = model.n_layers - 1
        pool = record_clean_pool(
            model, prompts, layers=LAYERS, target_layer=target_layer, max_seq_len=128, skip_first_n_positions=16
        )
        families = [EditFamily("gauss", size=size) for size in SIZES]
        pairs_by_family, stats = collect_pairs(
            model,
            prompts,
            pool,
            families=families,
            layers=LAYERS,
            target_layer=target_layer,
            pairs_per_prompt=8,
            rows_per_pass=128,
            max_seq_len=128,
            skip_first_n_positions=16,
        )
        print(f"== {dtype_name}  max_leak={stats.max_leak:.3g}", flush=True)
        for family in families:
            medians = []
            for layer in LAYERS:
                pairs = pairs_by_family[family.label][layer]
                first = pairs.prompt_idx_Int_P == 0
                errors_P = projected_jacobian_errors(
                    model,
                    prompts[0],
                    LayerPairs(*(getattr(pairs, name)[first] for name in LayerPairs.__dataclass_fields__)),
                    layer=layer,
                    target_layer=target_layer,
                    max_seq_len=128,
                    skip_first_n_positions=16,
                )
                medians.append(f"L{layer}={errors_P.median():.3f}")
            print(f"{dtype_name:9s} size {family.size:<7g} median jlens_err  " + "  ".join(medians), flush=True)
        del model, hf_model
        t.cuda.empty_cache()


if __name__ == "__main__":
    main()
