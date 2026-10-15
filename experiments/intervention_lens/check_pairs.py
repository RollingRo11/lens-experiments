"""Checks of collected intervention pairs on the real model (the dry run's checks).

For the first prompt with pairs, per family and layer, prints:

- ``jlens_err``: median / max relative error of each pair's summed target ``y`` against
  the J-lens estimator's first-order prediction ``G_s delta``
  (``projected_jacobian_errors``). Small for the gauss family means the hook edits the
  right row, block and position, and the right later positions are summed; for swaps
  it says how far a real-size edit is from linear.
- ``|y|/|delta|`` (median) and ``|y_same|/|y_sum|`` (median: how much of the summed
  effect is at the edited position itself).

Runs in the jpp_lens venv from the repo root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.intervention_fitting import (  # noqa: E402
    LayerPairs,
    PairsMeta,
    pairs_path,
    projected_jacobian_errors,
)
from workspace_lens.utils import resolve_target_layer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", required=True, help="The collect-pairs --out-dir.")
    parser.add_argument("--families", required=True, help="Comma-separated family labels.")
    args = parser.parse_args()

    meta = PairsMeta.load(args.pairs_dir)
    model = get_hf_model(meta.hf_model_name, attn_implementation="sdpa")
    target_layer = resolve_target_layer(model.n_layers, meta.relative_end_transport_layer)
    prompts = jpp_cli.load_fit_prompts(meta.num_prompts)
    print(f"model: n_layers={model.n_layers} d_model={model.d_model} target_layer={target_layer}")

    for label in args.families.split(","):
        for layer in meta.layers:
            pairs = LayerPairs.load(pairs_path(Path(args.pairs_dir) / label, layer))
            prompt_idx = int(pairs.prompt_idx_Int_P[0])
            from_prompt = pairs.prompt_idx_Int_P == prompt_idx
            prompt_pairs = LayerPairs(
                *(getattr(pairs, name)[from_prompt] for name in LayerPairs.__dataclass_fields__)
            )
            errors_P = projected_jacobian_errors(
                model,
                prompts[prompt_idx],
                prompt_pairs,
                layer=layer,
                target_layer=target_layer,
                max_seq_len=meta.max_seq_len,
                skip_first_n_positions=meta.skip_first_n_positions,
            )
            delta_norm_P = pairs.delta_PN.float().norm(dim=-1)
            y_sum_norm_P = pairs.y_sum_PF.float().norm(dim=-1)
            y_same_norm_P = pairs.y_same_PF.float().norm(dim=-1)
            print(
                f"{label:12s} layer {layer:2d}  pairs={len(pairs.delta_PN):6d}  "
                f"jlens_err median={errors_P.median():.3f} max={errors_P.max():.3f}  "
                f"|y|/|delta|={t.median(y_sum_norm_P / delta_norm_P):.2f}  "
                f"|y_same|/|y_sum|={t.median(y_same_norm_P / y_sum_norm_P):.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
