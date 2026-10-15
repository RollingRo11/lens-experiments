"""Add one model's column to the readout evals' correctness CSV.

Runs :meth:`ReadoutEvalRunner.grade_model_correctness` (one greedy step per item: is
the next token the item's target?) over every eval item and writes the input CSV
plus a ``--hf-model-name`` column to ``--out``. Pass that file to the pipeline's
``--correctness-csv`` so fit-weights and evaluate keep only the items the model
answers correctly.

    uv run python scripts/grade_correctness.py --hf-model-name google/gemma-4-E2B \
        --out artifacts/gemma4_e2b/model_correctness.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

import pandas as pd

from lens_evals.readout_evals.readout_eval_items import (
    MODEL_CORRECTNESS_CSV,
    READOUT_EVALS_DATA_DIR,
    load_readout_eval_items,
)
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model
from workspace_lens.utils import ensure_parent_dir

logger = logging.getLogger(__name__)

# The eval prompt truncation the pipeline uses (jpp_cli's RECIPE_READOUT_MAX_SEQ_LEN).
READOUT_MAX_SEQ_LEN = 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-model-name", required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--eval-data-dir", default=READOUT_EVALS_DATA_DIR)
    parser.add_argument(
        "--correctness-csv",
        default=MODEL_CORRECTNESS_CSV,
        help="CSV to add the column to. Default %(default)s.",
    )
    parser.add_argument("--max-seq-len", type=int, default=READOUT_MAX_SEQ_LEN)
    parser.add_argument(
        "--strip-trailing-space",
        action="store_true",
        help="Grade each prompt with trailing spaces removed. Gemma tokenizes a trailing "
        "space as its own token and then predicts markup, so every poetry item fails.",
    )
    parser.add_argument("--out", required=True, help="Output CSV (the input plus the new column).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    correctness_df = pd.read_csv(args.correctness_csv)
    if args.hf_model_name in correctness_df.columns:
        raise SystemExit(f"{args.correctness_csv} already has a {args.hf_model_name!r} column")

    model = get_hf_model(args.hf_model_name, attn_implementation=args.attn_implementation)
    items = load_readout_eval_items(args.eval_data_dir)
    if args.strip_trailing_space:
        # Grading only: the lens readouts still use the prompts as written.
        items = [dataclasses.replace(item, prompt=item.prompt.rstrip(" ")) for item in items]
    runner = ReadoutEvalRunner(model, {}, layers=[0], max_seq_len=args.max_seq_len)
    graded_df = runner.grade_model_correctness(items, hf_model_name=args.hf_model_name)

    merged_df = correctness_df.merge(graded_df, on=["eval", "item"], how="outer")
    ensure_parent_dir(args.out)
    merged_df.to_csv(args.out, index=False)

    column = graded_df[args.hf_model_name]
    for eval_slug, eval_column in column.groupby(graded_df["eval"]):
        graded = eval_column.dropna()
        if len(graded):
            logger.info("%s: %d/%d correct", eval_slug, int(graded.sum()), len(graded))
        else:
            logger.info("%s: ungradeable (%d items, always kept)", eval_slug, len(eval_column))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
