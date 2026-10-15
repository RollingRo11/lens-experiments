"""Are some lenses' recall@k hits ties at the top logit? The readout runner ranks a candidate as
1 + the number of strictly larger logits (ties go to the lens), on logits computed in the
LM head's dtype (bf16) after Gemma's tanh soft-cap, which flattens very large logits to the
same value. This rescores lenses with ties counted against the lens (rank = number of logits
>= the best candidate's) and exactly (fp32 logits before the soft-cap, which is monotone, so
the ranking is the true one), and reports per lens and layer how often the top logit is shared
by 10 or more tokens. Runs in the jpp_lens venv from the repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import sys

import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

import lens_evals.readout_evals.readout_evals as readout_evals_module  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402


def pessimistic_rank(logits_V: t.Tensor, candidate_ids) -> int:
    best_candidate_logit = logits_V[list(candidate_ids)].max()
    return int((logits_V >= best_candidate_logit).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--lens", nargs="+", required=True)
    parser.add_argument("--correctness-csv", required=True)
    parser.add_argument("--layers", default="4,8,12,16,20,24,28")
    parser.add_argument("--mode", choices=["pessimistic", "exact", "both"], default="both")
    parser.add_argument("--eval-data-dir", default=None, help="Default: the repo's readout evals.")
    parser.add_argument("--items", default="held-out:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    layers = [int(layer) for layer in args.layers.split(",")]
    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    lenses = jpp_cli.load_lenses(args.lens, model, hf_model_name=args.hf_model_name, layers=layers)
    _, items, excluded = jpp_cli.load_items_and_exclusions(
        model, eval_data_dir=args.eval_data_dir or jpp_cli.READOUT_EVALS_DATA_DIR,
        correctness_csv=args.correctness_csv, hf_model_name=args.hf_model_name, layers=layers,
        readout_max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN, items_spec=args.items,
    )

    # How often is the top logit shared by >= 10 tokens? One pass over the items per lens and layer.
    runner = jpp_cli.readout_runner(model, layers=layers, max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN)
    residuals = runner.record_residuals(items, cache_path=None, hf_model_name=args.hf_model_name)
    for name, lens in lenses.items():
        shares = {}
        for layer in layers:
            residual_PN = residuals.residuals_L_dict_PN[layer].to(model.input_device).float()
            logits_PV = model.unembed(lens.transport(residual_PN, layer)).float()
            ties_P = (logits_PV == logits_PV.max(dim=-1, keepdim=True).values).sum(dim=-1)
            shares[layer] = round(float((ties_P >= 10).float().mean()), 2)
        print(f"{name}: share of readouts whose top logit is shared by >= 10 tokens, by layer {shares}", flush=True)

    def report(label: str, out_path: str) -> None:
        ranks_df, pass_df = jpp_cli.evaluate(
            model, lenses, items, layers=layers, excluded_token_ids=excluded, ks=[1, 10]
        )
        ranks_df.to_csv(out_path, index=False)
        pass_df.to_csv(out_path.replace(".csv", "_pass_at_k.csv"), index=False)
        table = pass_df[pass_df["weighting"] == "item"]
        for k in (10, 1):
            print(f"recall@{k}, {label}")
            print((100 * table[table["k"] == k].pivot(index="lens", columns="eval", values="pass_at_k")).round(0).to_string(), flush=True)

    if args.mode in ("pessimistic", "both"):
        original_rank = readout_evals_module.min_rank_over_candidates
        readout_evals_module.min_rank_over_candidates = pessimistic_rank
        report("bf16 soft-capped logits, ties against the lens", args.out)
        readout_evals_module.min_rank_over_candidates = original_rank
    if args.mode in ("exact", "both"):
        # The exact ranking: fp32 logits before the (monotone) soft-cap, so nothing saturates.
        final_norm, unembed_VF = model._final_norm, model._lm_head.weight.float()

        def exact_unembed(residual):
            return final_norm(residual.float().to(unembed_VF.device)) @ unembed_VF.T

        model.unembed = exact_unembed
        report("fp32 logits before the soft-cap (exact ranks)", args.out.replace(".csv", "_exact.csv"))


if __name__ == "__main__":
    main()
