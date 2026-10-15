"""The plain-gradient J-lens at every layer (0-33 of Gemma 4 E2B) on the first 64 WikiText
prompts: the baseline for swaps over a band of middle layers, which J++ and the R-Lens (7
layers only) cannot make. Plain ``LensTrainer`` (standard gradients, no router). Runs in the
jpp_lens venv from the repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch as t

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402

from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.config import LensConfig  # noqa: E402
from workspace_lens.fitting.jacobian_fitting import LensTrainer  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
parser.add_argument("--num-prompts", type=int, default=64)
parser.add_argument("--rows-per-pass", type=int, default=32)
parser.add_argument("--artifacts-base-dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
config = LensConfig(
    hf_model_name=args.hf_model_name,
    checkpoint_name="plain_jlens64_all",
    artifacts_base_dir=args.artifacts_base_dir,
    source_layers=list(range(model.n_layers - 1)),
    jacobian_rows_per_pass=args.rows_per_pass,
    max_seq_len=jpp_cli.RECIPE_MAX_SEQ_LEN,
    skip_first_n_positions=jpp_cli.RECIPE_SKIP_FIRST_N_POSITIONS,
    checkpoint_every_n_prompts=None,
)
lens = LensTrainer(config, model, jpp_cli.load_fit_prompts(args.num_prompts)).fit()
lens.save(args.out, dtype=t.float32)
print("saved", args.out, f"peak GPU memory {t.cuda.max_memory_allocated() / 1e9:.1f} GB")
