"""The prompts of our J++ fitting split (``fit:0``, the items every fit-weights stage here
was fitted on), as a JSON list, for ``convert_banks.py --exclude-prompts``. Runs in the
jpp_lens venv from the repo root, on a GPU (the split is taken over the scoreable items)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from workspace_lens import get_hf_model  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
parser.add_argument("--correctness-csv", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
_, fitting_items, _ = jpp_cli.load_items_and_exclusions(
    model,
    eval_data_dir=jpp_cli.READOUT_EVALS_DATA_DIR,
    correctness_csv=args.correctness_csv,
    hf_model_name=args.hf_model_name,
    layers=[4, 8, 12, 16, 20, 24, 28],  # Gemma's band layers (RECIPE_LAYERS are Qwen's)
    readout_max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN,
    items_spec="fit:0",
)
Path(args.out).write_text(json.dumps([item.prompt for item in fitting_items]))
print(f"{len(fitting_items)} fitting-split prompts -> {args.out}")
