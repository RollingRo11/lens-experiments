"""WorkspaceBench readouts for our Gemma 4 E2B lenses, in the benchmark's readouts contract.

WorkspaceBench (github.com/camilablank/workspace-bench) judges one readouts JSONL per (lens,
family): rows ``{"id", "layer", "pos", "token", "tokens", "scores"}``, one per (item, layer,
read position). This script does what its bundled producer does, for our lenses and model:

- the read plan (``wsbench.readplan.plan``): per item, what to render, the positions rule and
  the layers. The plan's layers are Qwen3.6-27B's (64 blocks); ours are the lenses' own layers
  (default: the 7 band layers), so rows carry Gemma layer numbers and the judge is run with
  ``layers=`` set to them.
- the render (``wsbench.produce.render.render``) with Gemma's tokenizer and chat template; the
  positions rule is resolved on Gemma's tokens (``wsbench.readplan.resolve``), as the benchmark
  intends for another model.
- one forward per item, the block outputs at the read layers and positions, and for every lens
  the top-10 of ``unembed(A_l h)`` over the full vocabulary (our standard lens readout; the
  benchmark's own J-lens arm uses a cosine-normalised variant), token strings spelt as the
  benchmark spells them (``display_tokens``).

Families whose banks are Qwen-specific are not produced (``SKIPPED``). Items a chat render
cannot build with Gemma's template are skipped and counted. Runs in the jpp_lens venv with
``PYTHONPATH`` including the workspace-bench ``src/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402
from wsbench import readplan  # noqa: E402
from wsbench.produce.render import display_tokens, render  # noqa: E402

from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.residual_streams import residual_mean_over_streams  # noqa: E402

# Banks that are Qwen's own outputs, need re-generation or re-gating, or read at Qwen token
# indices (the benchmark's AGENTS.md, "Which families port").
SKIPPED = (
    "hallucination",
    "agentic_misalignment",
    "jlens_concept_pr",
    "directed_modulation",
    "jailbreak_recognition",
    "poetry",
    "moral_rationale",
    "brew_intermediates",  # its read positions are Qwen token indices
)
PORTABLE = (
    "association",
    "basic_readout",
    "multihop",
    "multilingual",
    "typo",
    "basic_readout_mt",
    "multihop_mt",
    "multilingual_mt",
    "typo_mt",
    "multilingual_multihop",
    "multilingual_typo",
    "chain_intermediates",
    "arithmetic_intermediates",
    "user_modeling",
    "conjunctive_association",
    "relational_multihop",
    "role_bound_association",
    "buggy_code",
    "multi_concept_directed_modulation",
)
TOP_K = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--families", default=",".join(PORTABLE))
    parser.add_argument("--layers", default="4,8,12,16,20,24,28")
    parser.add_argument("--lens", nargs="+", required=True, help="Lens .pt files (named by stem) or 'logit'.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Items rendering longer are skipped.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    layers = [int(layer) for layer in args.layers.split(",")]
    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    tokenizer = model.tokenizer
    lenses = jpp_cli.load_lenses(args.lens, model, hf_model_name=args.hf_model_name, layers=layers)
    for lens in lenses.values():
        lens.move_parameters_to_device(model.input_device)
    out_dir = Path(args.out_dir)
    for family in args.families.split(","):
        if family in SKIPPED:
            print(f"{family}: skipped (Qwen-specific bank)", flush=True)
            continue
        specs = readplan.plan(family)
        counts: Counter[str] = Counter()
        handles = {}
        for name in lenses:
            (out_dir / name).mkdir(parents=True, exist_ok=True)
            handles[name] = (out_dir / name / f"{family}.jsonl").open("w", encoding="utf-8")
        for spec in specs:
            if family == "basic_readout" and spec.render != "plain":
                counts["implicit item (Qwen's own favourite)"] += 1  # AGENTS.md: leave these out
                continue
            try:
                rendered = render(spec, tokenizer)
            except Exception as exc:  # a chat render Gemma's template rejects
                counts[f"render failed: {type(exc).__name__}"] += 1
                continue
            if len(rendered) > args.max_tokens:
                counts["too long"] += 1
                continue
            positions = readplan.resolve(spec.positions, rendered.tokens)
            if not positions:
                counts["no read position"] += 1
                continue
            indices = [position % len(rendered) for position in positions]
            input_ids_Int_1S = t.tensor([rendered.ids], device=model.input_device)
            with t.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward(input_ids_Int_1S)
                residuals_L_dict_PN = {
                    layer: residual_mean_over_streams(recorder.activations[layer])[0, indices].float()
                    for layer in layers
                }
            with t.no_grad():
                for name, lens in lenses.items():
                    rows = []
                    for layer in layers:
                        logits_PV = model.unembed(lens.transport(residuals_L_dict_PN[layer], layer)).float()
                        values_PK, ids_PK = logits_PV.topk(TOP_K, dim=-1)
                        for row_idx, (position, index) in enumerate(zip(positions, indices, strict=True)):
                            rows.append(
                                {
                                    "id": spec.id,
                                    "layer": layer,
                                    "pos": position,
                                    "token": rendered.decoded[index],
                                    "tokens": display_tokens(tokenizer, ids_PK[row_idx].tolist()),
                                    "scores": [round(v, 4) for v in values_PK[row_idx].tolist()],
                                }
                            )
                    handles[name].write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            counts["read"] += 1
        for handle in handles.values():
            handle.close()
        print(f"{family}: {len(specs)} items, {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
