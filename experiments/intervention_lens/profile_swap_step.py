"""Where does a swap-training step spend its time? On real rows of a swap dataset: the prompt
lengths, then for all layers and for a window of 3 layers, batch 256, the time of each part of a
step (building the batch, the clean residuals, the edited forward, the backward, the optimizer
step) with random rows, with rows of about one length (little padding), with the clean pass
stored up front (CleanResidualCache over every prompt of the dataset), and with the edit in bf16.
Runs in the jpp_lens venv from the repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from lens_evals.eval_utils import formatting_token_ids_of  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.swap_fitting import (  # noqa: E402
    CleanResidualCache,
    SwapBatch,
    SwapLoss,
)
from workspace_lens.utils import token_id_mask_Bool_V, vocab_size_of  # noqa: E402


def timed(fn):
    t.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    t.cuda.synchronize()
    return out, time.perf_counter() - start


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--swap-data", required=True)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--repeats", type=int, default=8)
args = parser.parse_args()

model = get_hf_model("google/gemma-4-E2B", attn_implementation="sdpa")
vocab_size = vocab_size_of(model)
excluded = token_id_mask_Bool_V(formatting_token_ids_of(model.tokenizer, vocab_size), vocab_size)
items = json.loads(Path(args.swap_data).read_text())["items"]
prompts, examples = jpp_cli.swap_examples_from_items(model, items, max_seq_len=128)
lengths = t.tensor([example.position + 1 for example in examples]).float()
print(f"{len(examples)} rows, {len(prompts)} prompts; length (tokens): mean {lengths.mean():.1f}, max {lengths.max():.0f}", flush=True)
ids_cache: dict[int, t.Tensor] = {}
all_layers = list(range(34))
(cache, build_s) = timed(
    lambda: CleanResidualCache(
        model, prompts, range(len(prompts)), layers=all_layers, max_seq_len=128, ids_cache=ids_cache
    )
)
print(f"clean cache: {build_s:.0f} s, {cache.store_LTN.numel() * cache.store_LTN.element_size() / 1e9:.1f} GB", flush=True)
rng = random.Random(1)
by_length = sorted(range(len(examples)), key=lambda i: examples[i].position)


def draw_rows(one_length: bool) -> list:
    if not one_length:
        return [examples[i] for i in rng.sample(range(len(examples)), args.batch_size)]
    start = rng.randrange(len(examples) - args.batch_size)
    return [examples[i] for i in by_length[start : start + args.batch_size]]


def time_steps(name: str, layers: list[int], *, one_length: bool, cached: bool, edit_dtype: t.dtype | None) -> None:
    maps = {layer: t.eye(model.d_model, device="cuda").requires_grad_(True) for layer in layers}
    optimizer = t.optim.Adam(list(maps.values()), lr=1e-4)
    swap_loss = SwapLoss(
        model, layers=layers, mode="supervised", scale=1.0, window=1, excluded_mask_Bool_V=excluded,
        edit_dtype=edit_dtype, clean_cache=cache if cached else None,
    )
    parts = dict.fromkeys(["build batch", "clean residuals", "edited forward", "backward", "optimizer"], 0.0)
    t.cuda.reset_peak_memory_stats()
    for repeat in range(args.repeats + 1):  # the first repeat warms up and is not counted
        rows = draw_rows(one_length)
        batch, build = timed(lambda: SwapBatch.build(model, prompts, rows, max_seq_len=128, ids_cache=ids_cache))  # noqa: B023
        _, clean = timed(lambda: swap_loss._targets(batch, layers))  # noqa: B023
        (loss, _), clean_and_edited = timed(lambda: swap_loss(maps, batch))  # noqa: B023
        optimizer.zero_grad(set_to_none=True)
        _, backward = timed(lambda: loss.backward())  # noqa: B023
        _, step = timed(optimizer.step)
        if repeat:
            for part, value in zip(parts, [build, clean, clean_and_edited - clean, backward, step], strict=True):
                parts[part] += value / args.repeats
    step_s = sum(parts.values())
    print(
        f"{name:58s}: {step_s:.3f} s/step, {args.batch_size / step_s:5.0f} rows/s | "
        + ", ".join(f"{part} {100 * value / step_s:.0f}%" for part, value in parts.items())
        + f" | peak {t.cuda.max_memory_allocated() / 1e9:.1f} GB",
        flush=True,
    )


for layer_set_name, layers in [("all 34 layers", all_layers), ("window of 3", [16, 17, 18])]:
    time_steps(f"{layer_set_name}, random rows", layers, one_length=False, cached=False, edit_dtype=t.float32)
    time_steps(f"{layer_set_name}, one length", layers, one_length=True, cached=False, edit_dtype=t.float32)
    time_steps(f"{layer_set_name}, one length, stored clean pass", layers, one_length=True, cached=True, edit_dtype=t.float32)
    time_steps(f"{layer_set_name}, one length, stored clean pass, bf16 edit", layers, one_length=True, cached=True, edit_dtype=None)
