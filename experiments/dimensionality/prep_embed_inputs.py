"""Inputs for embed_layer.py, written as plain numpy so that step can run in an env
without torch: the same random 20k rows embed_2d.py used, their 64-dim router
projection and K=8 router cluster per layer, and a hover label per row (the token
with the 10 tokens before it, from re-tokenising the prompt as the model's
``encode`` does, checked against the saved token ids).

Run with the project venv (torch + transformers):
    .venv/bin/python -u experiments/dimensionality/prep_embed_inputs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as t
from transformers import AutoTokenizer

from workspace_lens.routing.router import ActivationRouterCollection

CONTEXT_TOKENS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", default="artifacts/gemma4_e2b/activations")
    parser.add_argument("--router-path", default="artifacts/gemma4_e2b/router.pt")
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28])
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/dimensionality/embed_inputs.npz")
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    router = ActivationRouterCollection.load(args.router_path)
    index = t.load(activations_dir / "index.pt", weights_only=True)
    total_points = index["position"].shape[0]
    # Same rows as embed_2d.py.
    rows_Int_S = t.randperm(total_points, generator=t.Generator().manual_seed(args.seed))[
        : args.num_points
    ].sort().values

    prompt_idx_S = index["prompt_idx"][rows_Int_S]
    position_S = index["position"][rows_Int_S]
    token_id_S = index["token_id"][rows_Int_S]
    saved: dict[str, np.ndarray] = {
        "rows": rows_Int_S.numpy(),
        "prompt_idx": prompt_idx_S.numpy(),
        "position": position_S.numpy(),
        "token_id": token_id_S.numpy(),
        "layers": np.array(args.layers),
    }

    prompts = json.loads((activations_dir / "prompts.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    ids_by_prompt: dict[int, list[int]] = {}
    labels: list[str] = []
    for prompt_idx, position, token_id in zip(
        prompt_idx_S.tolist(), position_S.tolist(), token_id_S.tolist()
    ):
        if prompt_idx not in ids_by_prompt:
            ids_by_prompt[prompt_idx] = tokenizer(
                prompts[prompt_idx], truncation=True, max_length=args.max_seq_len
            ).input_ids
        ids = ids_by_prompt[prompt_idx]
        assert ids[position] == token_id, (prompt_idx, position)
        before = tokenizer.decode(ids[max(0, position - CONTEXT_TOKENS) : position])
        labels.append(f"…{before}⟦{tokenizer.decode([ids[position]])}⟧")
    saved["label"] = np.array(labels)

    for layer in args.layers:
        projected_SD = t.load(
            activations_dir / f"proj{router.projection_dim}_layer{layer}.pt", weights_only=True
        )[rows_Int_S]
        centroids_ED = router.layer_routers_L_dict[layer].centroids_ED
        saved[f"proj_{layer}"] = projected_SD.numpy().astype(np.float32)
        saved[f"cluster_{layer}"] = t.cdist(projected_SD, centroids_ED).argmin(dim=1).numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **saved)
    print(f"saved {out_path}: {args.num_points} rows, labels checked against saved token ids")


if __name__ == "__main__":
    main()
