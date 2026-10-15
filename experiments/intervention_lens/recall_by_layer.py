"""Recall@k of each lens restricted to each single layer, every window of 3 and of 5 consecutive
layers, and all layers, from an ``evaluate`` ranks CSV (the README metric: per item, the share
of intermediates whose best rank over the chosen layers is within k; ``macro`` = mean over
the five evals). More layers give more chances, so compare lenses on the same layer set.
Runs in the jpp_lens venv from the repo root.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

sys.path.insert(0, "src")

from lens_evals.readout_evals.readout_eval_scoring import pass_at_k_over_layers  # noqa: E402
from workspace_lens.fitting.swap_fitting import layer_windows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", required=True, help="evaluate's ranks CSV.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--window-sizes", default="1,3,5")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ranks_df = pd.read_csv(args.ranks)
    sizes = [int(size) for size in args.window_sizes.split(",")]
    rows = []
    for lens, lens_ranks_df in ranks_df.groupby("lens"):
        for name, layers in layer_windows(sorted(lens_ranks_df["layer"].unique()), sizes).items():
            pass_df = pass_at_k_over_layers(lens_ranks_df[lens_ranks_df["layer"].isin(layers)], [args.k])
            for row in pass_df.itertuples():
                rows.append({"lens": lens, "layers": name, "eval": row.eval, f"recall_at_{args.k}": row.pass_at_k})
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    macro = table[table["eval"] == "macro"].pivot(index="layers", columns="lens", values=f"recall_at_{args.k}")
    print((100 * macro).round(0).to_string())


if __name__ == "__main__":
    main()
