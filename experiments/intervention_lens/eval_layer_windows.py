"""Swap success of lenses with the swap made at each single layer, at every window of 3 and
of 5 consecutive layers, at all layers, optionally over a band, and with no edit at all
(``none``: the share of items whose swapped-in answer Gemma already gives), on one or more
swap datasets (probe-swap's edit restricted to those layers; ``swap_hits_by_layer_set``).

Writes one row per (lens, dataset, layer set, item) to ``--out`` (parquet, or CSV for a
``.csv`` path) with the item's relation, canonical relation ID and clean ranks, and a summary
beside it (``<stem>_summary.csv``: item-level hit rate and the mean over relations of each
relation's hit rate). ``select_windows.py`` picks windows on one dataset and reports them on
another. Runs in the jpp_lens venv from the repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from lens_evals.eval_utils import formatting_token_ids_of  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.swap_fitting import (  # noqa: E402
    layer_windows,
    swap_hits_by_layer_set,
)
from workspace_lens.utils import load_lens_file, token_id_mask_Bool_V, vocab_size_of  # noqa: E402

ITEM_COLUMNS = ("relation", "relation_id", "clean_answer_rank", "swap_answer_clean_rank")


def parse_data_spec(spec: str) -> tuple[str, str]:
    """``[name=]path``; the name defaults to the file stem."""
    name, separator, path = spec.partition("=")
    return (name, path) if separator else (Path(spec).stem, spec)


def summarise(rows_df: pd.DataFrame) -> pd.DataFrame:
    """Per (lens, dataset, layers): items, relations, item-level hit rate and the mean over
    relations (``relation_id``) of each relation's hit rate."""
    groups = ["lens", "dataset", "layers"]
    per_relation = rows_df.groupby([*groups, "relation_id"])["hit"].mean().rename("relation_hit_rate").reset_index()
    summary = rows_df.groupby(groups).agg(
        items=("hit", "size"), hit_rate=("hit", "mean"), loss=("loss", "first")
    ).reset_index()
    relation_means = per_relation.groupby(groups).agg(
        relations=("relation_hit_rate", "size"), relation_mean=("relation_hit_rate", "mean")
    ).reset_index()
    return summary.merge(relation_means, on=groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--lens", nargs="+", required=True, help="Lens .pt files, or 'logit:<layers>'.")
    parser.add_argument("--data", nargs="+", required=True, help="Swap datasets in the probe-swap format, [name=]path.")
    parser.add_argument("--window-sizes", default="1,3,5")
    parser.add_argument(
        "--band", default=None, help="e.g. 8-26: also swap at every layer from 8 to 26 at once (lenses that have them)."
    )
    parser.add_argument("--max-items", type=int, default=None, help="First N items of each dataset.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    vocab_size = vocab_size_of(model)
    excluded_mask_Bool_V = token_id_mask_Bool_V(formatting_token_ids_of(model.tokenizer, vocab_size), vocab_size)
    datasets = {}
    for spec in args.data:
        name, path = parse_data_spec(spec)
        items = json.loads(Path(path).read_text())["items"][: args.max_items]
        prompts, examples, kept = jpp_cli.swap_examples_and_kept_items(model, items, max_seq_len=128)
        datasets[name] = (prompts, examples, [items[idx] for idx in kept])
        print(f"{name}: {len(examples)} of {len(items)} items scoreable", flush=True)
    sizes = [int(size) for size in args.window_sizes.split(",")]
    frames = []
    for spec in args.lens:
        if spec.startswith("logit:"):
            lens_name, layers = "logit", [int(layer) for layer in spec.split(":", 1)[1].split(",")]
            maps_L_dict_FN = {layer: t.eye(model.d_model) for layer in layers}
        else:
            lens = load_lens_file(spec, hf_model_name=args.hf_model_name)
            lens_name, maps_L_dict_FN = Path(spec).stem, lens.jacobians_L_dict_FN
        layer_sets = layer_windows(list(maps_L_dict_FN), sizes)
        if args.band:
            first, last = (int(layer) for layer in args.band.split("-"))
            if all(layer in maps_L_dict_FN for layer in range(first, last + 1)):
                layer_sets[f"band{first}-{last}"] = list(range(first, last + 1))
        layer_sets["none"] = []
        for dataset_name, (prompts, examples, items) in datasets.items():
            results = swap_hits_by_layer_set(
                model, prompts, examples, maps_L_dict_FN, layer_sets, excluded_mask_Bool_V=excluded_mask_Bool_V,
            )
            for window, (loss, hits_Bool_N) in results.items():
                frame = pd.DataFrame({column: [item.get(column) for item in items] for column in ITEM_COLUMNS})
                frame.insert(0, "item", [item["name"] for item in items])
                frame["relation"] = frame["relation"].fillna(pd.Series([item.get("category") for item in items]))
                frame["relation_id"] = frame["relation_id"].fillna(frame["relation"])
                frame = frame.assign(
                    lens=lens_name, dataset=dataset_name, layers=window, loss=loss, hit=hits_Bool_N.numpy()
                )
                frames.append(frame)
            rates = {window: round(float(hits.float().mean()), 3) for window, (_, hits) in results.items()}
            print(lens_name, dataset_name, rates, flush=True)
    rows_df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    if out.suffix == ".csv":
        rows_df.to_csv(out, index=False)
    else:
        rows_df.to_parquet(out, index=False)
    summary_path = out.with_name(f"{out.stem}_summary.csv")
    summarise(rows_df).to_csv(summary_path, index=False)
    print("wrote", out, "and", summary_path, flush=True)


if __name__ == "__main__":
    main()
