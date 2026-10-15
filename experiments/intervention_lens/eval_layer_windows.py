"""Swap success of lenses with the swap made at each single layer, at every window of 3 and
of 5 consecutive layers, and at all layers, on a swap dataset (probe-swap's edit restricted
to those layers; ``swap_success_by_layer_set``). Writes one CSV row per (lens, layer set).
Runs in the jpp_lens venv from the repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from lens_evals.eval_utils import formatting_token_ids_of  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.swap_fitting import (  # noqa: E402
    layer_windows,
    swap_success_by_layer_set,
)
from workspace_lens.utils import load_lens_file, token_id_mask_Bool_V, vocab_size_of  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--lens", nargs="+", required=True, help="Lens .pt files, or 'logit:<layers>'.")
    parser.add_argument("--data", required=True, help="Swap items in the probe-swap format.")
    parser.add_argument("--window-sizes", default="1,3,5")
    parser.add_argument(
        "--band", default=None, help="e.g. 8-26: also swap at every layer from 8 to 26 at once (lenses that have them)."
    )
    parser.add_argument("--max-items", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    vocab_size = vocab_size_of(model)
    excluded_mask_Bool_V = token_id_mask_Bool_V(formatting_token_ids_of(model.tokenizer, vocab_size), vocab_size)
    items = json.loads(Path(args.data).read_text())["items"][: args.max_items]
    prompts, examples = jpp_cli.swap_examples_from_items(model, items, max_seq_len=128)
    sizes = [int(size) for size in args.window_sizes.split(",")]
    rows = []
    for spec in args.lens:
        if spec.startswith("logit:"):
            name, layers = "logit", [int(layer) for layer in spec.split(":", 1)[1].split(",")]
            maps_L_dict_FN = {layer: t.eye(model.d_model) for layer in layers}
        else:
            lens = load_lens_file(spec, hf_model_name=args.hf_model_name)
            name, maps_L_dict_FN = Path(spec).stem, lens.jacobians_L_dict_FN
        layer_sets = layer_windows(list(maps_L_dict_FN), sizes)
        if args.band:
            first, last = (int(layer) for layer in args.band.split("-"))
            if all(layer in maps_L_dict_FN for layer in range(first, last + 1)):
                layer_sets[f"band{first}-{last}"] = list(range(first, last + 1))
        results = swap_success_by_layer_set(
            model, prompts, examples, maps_L_dict_FN, layer_sets,
            excluded_mask_Bool_V=excluded_mask_Bool_V,
        )
        for window, (loss, hit_rate) in results.items():
            rows.append({"lens": name, "layers": window, "loss": loss, "hit_rate": hit_rate})
        print(name, {window: round(hit_rate, 3) for window, (_, hit_rate) in results.items()}, flush=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lens", "layers", "loss", "hit_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out, f"({len(examples)} swaps)")


if __name__ == "__main__":
    main()
