"""Pick each lens's best layer window on one swap dataset (dev) and report it on another
(test), from the per-item table of ``eval_layer_windows.py``.

For every lens and window size (1, 3, 5 consecutive layers) the window with the highest
relation-level mean on dev is chosen; on test the table gives that window's relation-level
mean (each relation's hit rate, averaged over relations, so a large relation does not
dominate), its item-level rate, and a 95% bootstrap interval over test relations. "all",
the band and "none" (no edit: the share of items whose swapped-in answer Gemma already
gives) need no choosing and are reported as they are. With ``--clean-top1`` only items whose
own answer is Gemma's top word on the clean prompt are counted. Runs anywhere with pandas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def relation_rates(rows_df: pd.DataFrame) -> pd.Series:
    """Hit rate per relation (``relation_id``)."""
    return rows_df.groupby("relation_id")["hit"].mean()


def bootstrap_interval(rates: pd.Series, *, draws: int, seed: int) -> tuple[float, float]:
    """95% interval of the mean over relations, resampling relations."""
    values = rates.to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    means = values[generator.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def window_size(name: str) -> int | None:
    """Layers in a window named ``L<a>`` or ``L<a>-<b>``; None for all, bands and none."""
    if not name.startswith("L"):
        return None
    first, _, last = name[1:].partition("-")
    return int(last or first) - int(first) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="Per-item table from eval_layer_windows.py.")
    parser.add_argument("--select-on", default="dev")
    parser.add_argument("--report-on", default="test")
    parser.add_argument("--clean-top1", action="store_true", help="Only items whose own answer is the top word.")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = Path(args.rows)
    rows_df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if args.clean_top1:
        rows_df = rows_df[rows_df["clean_answer_rank"] == 0]
    out_rows = []
    for lens, lens_df in rows_df.groupby("lens", sort=False):
        select_df = lens_df[lens_df["dataset"] == args.select_on]
        report_df = lens_df[lens_df["dataset"] == args.report_on]
        choices: list[tuple[str, str]] = []
        names = list(dict.fromkeys(report_df["layers"]))
        for size in sorted({window_size(name) for name in names} - {None}):
            windows = [name for name in names if window_size(name) == size]
            dev_means = {window: relation_rates(select_df[select_df["layers"] == window]).mean() for window in windows}
            best = max(windows, key=lambda window: dev_means[window])
            choices.append((f"best window of {size} (chosen on {args.select_on})", best))
        choices += [(name, name) for name in names if window_size(name) is None]
        for label, window in choices:
            window_df = report_df[report_df["layers"] == window]
            rates = relation_rates(window_df)
            low, high = bootstrap_interval(rates, draws=args.draws, seed=args.seed)
            dev_rates = relation_rates(select_df[select_df["layers"] == window])
            out_rows.append({
                "lens": lens, "choice": label, "layers": window,
                f"{args.select_on}_relation_mean": float(dev_rates.mean()) if len(dev_rates) else float("nan"),
                f"{args.report_on}_relation_mean": float(rates.mean()),
                f"{args.report_on}_relation_mean_low": low,
                f"{args.report_on}_relation_mean_high": high,
                f"{args.report_on}_item_rate": float(window_df["hit"].mean()),
                f"{args.report_on}_items": len(window_df),
                f"{args.report_on}_relations": len(rates),
            })
    table = pd.DataFrame(out_rows)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
