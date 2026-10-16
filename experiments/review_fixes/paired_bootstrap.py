"""Bootstrap intervals for recall@k from per-question ranks, for single lenses and for paired
differences between lenses scored on the same questions.

Reads the rank rows that ``jpp_cli.py evaluate`` writes (one row per lens, item, intermediate
and layer). A question's score is the share of its intermediates whose best rank over the
layers is <= k; a task's recall@k is the mean over its questions; the macro is the mean over
the five tasks (``pass_at_k_over_layers``). Each bootstrap draw resamples the questions of
every task with replacement, and the same draw is used for every lens, so a difference
between two lenses is paired: it only moves when the questions on which they differ are
drawn. Reports the point value and a 95% interval per lens and task, and per lens pair and
task (including the macro). The intervals cover the choice of questions only, not the noise
of fitting a lens. Runs anywhere with pandas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS  # noqa: E402


def question_scores(ranks_df: pd.DataFrame, *, k: int) -> pd.DataFrame:
    """``[question, lens]`` table of question scores, indexed by (eval, item)."""
    best_rank = ranks_df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min()
    per_question = (best_rank <= k).groupby(["eval", "lens", "item"]).mean()
    return per_question.unstack("lens")


def bootstrap(
    scores_df: pd.DataFrame, *, evals: list[str], draws: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``(point, samples)``: per task and ``macro``, the ``[lens]`` point values and the
    ``[draw, lens]`` bootstrap values (same resampled questions for every lens)."""
    generator = np.random.default_rng(seed)
    point, samples = {}, {}
    for eval_slug in evals:
        values_QL = scores_df.loc[eval_slug].to_numpy(dtype=float)
        point[eval_slug] = values_QL.mean(axis=0)
        picks_DQ = generator.integers(0, len(values_QL), size=(draws, len(values_QL)))
        samples[eval_slug] = values_QL[picks_DQ].mean(axis=1)
    point["macro"] = np.mean([point[eval_slug] for eval_slug in evals], axis=0)
    samples["macro"] = np.mean([samples[eval_slug] for eval_slug in evals], axis=0)
    return point, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", nargs="+", required=True, help="Rank CSVs from evaluate (lens names must be unique).")
    parser.add_argument("--lens", nargs="*", default=None, help="Lenses to keep (default: all).")
    parser.add_argument("--pairs", nargs="*", default=[], help="Paired differences, each as A-minus-B written A:B.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="CSV of every interval; <stem>_pairs.csv holds the pairs.")
    args = parser.parse_args()

    ranks_df = pd.concat([pd.read_csv(path) for path in args.ranks], ignore_index=True)
    ranks_df = ranks_df.drop_duplicates(["eval", "lens", "item", "intermediate", "layer"])
    if args.lens:
        missing = sorted(set(args.lens) - set(ranks_df["lens"]))
        if missing:
            raise SystemExit(f"lenses not in the rank files: {missing}")
        ranks_df = ranks_df[ranks_df["lens"].isin(args.lens)]
    scores_df = question_scores(ranks_df, k=args.k)
    # Only questions every lens was scored on, so every difference is paired.
    incomplete = scores_df.isna().any(axis=1)
    if incomplete.any():
        print(f"dropping {int(incomplete.sum())} questions not scored for every lens", flush=True)
        scores_df = scores_df[~incomplete]
    evals = [eval_slug for eval_slug in MACRO_EVALS if eval_slug in scores_df.index.get_level_values("eval")]
    if len(evals) != len(MACRO_EVALS):
        print(f"warning: only {evals} present; the macro averages those", flush=True)
    lenses = list(scores_df.columns)
    point, samples = bootstrap(scores_df, evals=evals, draws=args.draws, seed=args.seed)
    questions = {eval_slug: int(len(scores_df.loc[eval_slug])) for eval_slug in evals}
    print(f"questions per task: {questions}; {args.draws} draws; recall@{args.k} in points (x100)", flush=True)

    rows = []
    for lens_idx, lens in enumerate(lenses):
        for task in [*evals, "macro"]:
            low, high = np.percentile(samples[task][:, lens_idx], [2.5, 97.5])
            rows.append({"lens": lens, "eval": task, "recall": 100 * point[task][lens_idx],
                         "low": 100 * low, "high": 100 * high})
    lens_table = pd.DataFrame(rows)
    lens_table.to_csv(args.out, index=False)
    wide = lens_table[lens_table["eval"] == "macro"].set_index("lens")[["recall", "low", "high"]]
    print(wide.sort_values("recall", ascending=False).round(1).to_string(), flush=True)

    pair_rows = []
    for pair in args.pairs:
        first, _, second = pair.partition(":")
        if first not in lenses or second not in lenses:
            print(f"skipping pair {pair}: lens not found", flush=True)
            continue
        a, b = lenses.index(first), lenses.index(second)
        for task in [*evals, "macro"]:
            diff_D = samples[task][:, a] - samples[task][:, b]
            low, high = np.percentile(diff_D, [2.5, 97.5])
            pair_rows.append({
                "lens_a": first, "lens_b": second, "eval": task,
                "difference": 100 * (point[task][a] - point[task][b]), "low": 100 * low, "high": 100 * high,
                "share_of_draws_a_not_above_b": float((diff_D <= 0).mean()),
                "questions": questions.get(task, sum(questions.values())),
            })
    if pair_rows:
        pair_table = pd.DataFrame(pair_rows)
        pair_path = Path(args.out).with_name(f"{Path(args.out).stem}_pairs.csv")
        pair_table.to_csv(pair_path, index=False)
        with pd.option_context("display.width", 200):
            print(pair_table.round(2).to_string(index=False), flush=True)
        print("wrote", args.out, "and", pair_path, flush=True)
    else:
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
