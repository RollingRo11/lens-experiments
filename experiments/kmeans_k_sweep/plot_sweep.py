"""Plot the k-means loss vs K from extract_and_sweep.py's sweep.json: one panel per layer.

Run with matplotlib layered on the project env:
    uv run --with matplotlib python experiments/kmeans_k_sweep/plot_sweep.py SWEEP_JSON OUT_PNG
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt

LINE = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(sweep_path: str, out_path: str) -> None:
    sweep = json.loads(open(sweep_path).read())
    # K=1 (one centre at the mean: the points' total spread) anchors each curve.
    ks = [1] + sweep["ks"]
    num_points = sweep["num_points"]
    layers = sorted(sweep["layers"], key=int)

    ncols = 4
    nrows = -(-len(layers) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), sharex=True, sharey=True,
        squeeze=False,
    )
    for ax in axes.flat[len(layers):]:
        ax.set_visible(False)

    for ax, layer in zip(axes.flat, layers):
        # Mean squared distance to the nearest centre: inertia / number of points.
        layer_sweep = sweep["layers"][layer]
        per_seed = [[layer_sweep["k1_inertia"] / num_points]] + [
            [v / num_points for v in layer_sweep["inertia"][str(k)]] for k in ks[1:]
        ]
        mean = [sum(v) / len(v) for v in per_seed]
        ax.fill_between(ks, [min(v) for v in per_seed], [max(v) for v in per_seed],
                        color=LINE, alpha=0.2, linewidth=0)
        ax.plot(ks, mean, color=LINE, linewidth=2, marker="o", markersize=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_title(f"layer {layer}", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    # Shared y from 0 to the largest loss (K=1 is the largest in every layer).
    top = max(sweep["layers"][layer]["k1_inertia"] for layer in layers) / num_points
    axes[0, 0].set_ylim(0, 1.05 * top)
    for ax in axes[-1]:
        ax.set_xlabel("K (number of clusters)", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("k-means loss\n(mean sq. distance)", color=MUTED, fontsize=9)
    # A hidden bottom-row panel leaves the panel above it without a K label.
    for ax in axes.flat[len(layers) - ncols:len(layers)]:
        ax.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
