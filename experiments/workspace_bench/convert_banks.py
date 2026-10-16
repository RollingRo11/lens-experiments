"""WorkspaceBench's single-token families as our readout-eval files, scored by our recall@k
instead of the benchmark's LLM judge.

The banks of association, multihop, multilingual, poetry and typo
(``<workspace-bench>/evals/<family>/items.json``) carry the fields of our
``lens-eval-<slug>.json`` items (``name``, ``prompt``, ``target``, ``intermediates``). Writes
``lens-eval-<slug>.json`` for each into ``--out-dir``, dropping every item whose prompt is also
a prompt of our own question files (``--ours``: every item of every ``lens-eval-*.json``, not
only the J++ fitting split), compared after collapsing whitespace and ignoring case, so no
lens is scored here on a question it was fitted or scored on before. ``--exclude-prompts``
(a JSON list) drops more prompts the same way. Prints, per family, the bank size, the exact
copies of our prompts, the copies after normalising, and the items kept.
``lens-eval-order-ops.json`` is copied from ours (the correctness grader loads every eval).
Also writes ``correctness_seed.csv`` (header only) for ``scripts/grade_correctness.py`` and
``overlap.csv`` (the per-family counts).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

FAMILIES = ("association", "multihop", "multilingual", "poetry", "typo")
FIELDS = ("name", "prompt", "target", "intermediates")


def normalise(prompt: str) -> str:
    """The prompt with runs of whitespace collapsed to one space, ends stripped, case folded."""
    return " ".join(prompt.split()).casefold()


def our_prompts(ours: Path) -> list[str]:
    """Every prompt of every ``lens-eval-*.json`` in ``ours``."""
    prompts = []
    for path in sorted(ours.glob("lens-eval-*.json")):
        payload = json.loads(path.read_text())
        items = payload["items"] if isinstance(payload, dict) else payload
        prompts += [item["prompt"] for item in items]
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsbench", required=True, help="The workspace-bench checkout.")
    parser.add_argument("--ours", default="data/jlens/evaluations", help="Our eval data dir.")
    parser.add_argument("--exclude-prompts", default=None, help="JSON list of more prompts to drop.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ours = Path(args.ours)
    ours_exact = set(our_prompts(ours))
    extra = json.loads(Path(args.exclude_prompts).read_text()) if args.exclude_prompts else []
    excluded_normalised = {normalise(prompt) for prompt in [*ours_exact, *extra]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap_rows = []
    for family in FAMILIES:
        bank = json.loads((Path(args.wsbench) / "evals" / family / "items.json").read_text())
        items = bank["items"] if isinstance(bank, dict) else bank
        exact = sum(item["prompt"] in ours_exact for item in items)
        normalised = sum(normalise(item["prompt"]) in excluded_normalised for item in items)
        kept = [
            {field: item[field] for field in FIELDS if field in item}
            for item in items
            if normalise(item["prompt"]) not in excluded_normalised and item.get("intermediates")
        ]
        (out_dir / f"lens-eval-{family}.json").write_text(json.dumps({"items": kept}, ensure_ascii=False, indent=1))
        overlap_rows.append(
            {"family": family, "bank": len(items), "exact_copies": exact, "normalised_copies": normalised,
             "kept": len(kept)}
        )
        print(f"{family}: {len(items)} items, {exact} exact copies of our prompts, {normalised} after "
              f"normalising (or in --exclude-prompts); {len(kept)} kept", flush=True)
    with open(out_dir / "overlap.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlap_rows[0]))
        writer.writeheader()
        writer.writerows(overlap_rows)
    shutil.copy(ours / "lens-eval-order-ops.json", out_dir / "lens-eval-order-ops.json")
    (out_dir / "correctness_seed.csv").write_text("eval,item\n")


if __name__ == "__main__":
    main()
