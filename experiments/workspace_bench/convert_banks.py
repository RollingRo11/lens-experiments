"""WorkspaceBench's single-token families as our readout-eval files, scored by our recall@k
instead of the benchmark's LLM judge.

The banks of association, multihop, multilingual, poetry and typo
(``<workspace-bench>/evals/<family>/items.json``) carry the fields of our
``lens-eval-<slug>.json`` items (``name``, ``prompt``, ``target``, ``intermediates``). Writes
``lens-eval-<slug>.json`` for each into ``--out-dir``, dropping the items whose prompt is in
``--exclude-prompts`` (a JSON list: the prompts of our J++ fitting split, so no lens with
fitted expert weights is scored on an item it was fitted on). ``lens-eval-order-ops.json`` is
copied from ours (the correctness grader loads every eval). Also writes ``correctness_seed.csv`` (header only) for
``scripts/grade_correctness.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FAMILIES = ("association", "multihop", "multilingual", "poetry", "typo")
FIELDS = ("name", "prompt", "target", "intermediates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsbench", required=True, help="The workspace-bench checkout.")
    parser.add_argument("--ours", default="data/jlens/evaluations", help="Our eval data dir.")
    parser.add_argument("--exclude-prompts", default=None, help="JSON list of prompts to drop.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ours = Path(args.ours)
    excluded = set(json.loads(Path(args.exclude_prompts).read_text())) if args.exclude_prompts else set()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        bank = json.loads((Path(args.wsbench) / "evals" / family / "items.json").read_text())
        items = bank["items"] if isinstance(bank, dict) else bank
        kept = [
            {field: item[field] for field in FIELDS if field in item}
            for item in items
            if item["prompt"] not in excluded and item.get("intermediates")
        ]
        (out_dir / f"lens-eval-{family}.json").write_text(json.dumps({"items": kept}, ensure_ascii=False, indent=1))
        print(f"{family}: {len(kept)} of {len(items)} items kept ({len(items) - len(kept)} in the J++ fitting split or empty)")
    shutil.copy(ours / "lens-eval-order-ops.json", out_dir / "lens-eval-order-ops.json")
    (out_dir / "correctness_seed.csv").write_text("eval,item\n")


if __name__ == "__main__":
    main()
