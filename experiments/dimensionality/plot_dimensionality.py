"""Plots for pca_and_twonn.json and the per-layer embeddings from embed_layer.py.

    uv run --with matplotlib python experiments/dimensionality/plot_dimensionality.py \
        results/dimensionality/pca_and_twonn.json results/dimensionality/embed_inputs.npz \
        results/dimensionality/embed_3d OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ordinal blue ramp, shallow (light) to deep (dark) layers.
LAYER_COLORS = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_variance_and_twonn(pca_path: str, out_path: Path) -> None:
    layers_data = json.loads(open(pca_path).read())["layers"]
    layers = sorted(layers_data, key=int)
    fig, (ax_var, ax_id) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})

    for color, layer in zip(LAYER_COLORS, layers):
        eigenvalues = np.array(layers_data[layer]["eigenvalues"])
        cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
        dims = np.arange(1, len(cumulative) + 1)
        ax_var.plot(dims, cumulative, color=color, linewidth=2, label=f"layer {layer}")
    ax_var.axvline(64, color=MUTED, linestyle="--", linewidth=1)
    ax_var.text(64 * 1.08, 0.03, "64 (router)", color=MUTED, fontsize=8)
    ax_var.set_xscale("log", base=2)
    ax_var.set_xlim(1, len(cumulative))
    ticks = [1, 4, 16, 64, 256, 1024]
    ax_var.set_xticks(ticks, [str(tick) for tick in ticks])
    ax_var.minorticks_off()
    ax_var.set_ylim(0, 1)
    ax_var.set_xlabel("number of PCA dims kept", color=MUTED, fontsize=9)
    ax_var.set_ylabel("fraction of variance explained", color=MUTED, fontsize=9)
    ax_var.set_title("PCA: variance explained", color=INK, fontsize=10)
    ax_var.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
    style(ax_var)

    layer_ints = [int(layer) for layer in layers]
    ax_id.plot(
        layer_ints, [layers_data[layer]["twonn"]["all"]["dimension"] for layer in layers],
        color=LAYER_COLORS[3], linewidth=2, marker="o", markersize=5, label="all points",
    )
    subsets = [name for name in layers_data[layers[0]]["twonn"] if name != "all"]
    for marker, name in zip(["^", "s"], subsets):
        ax_id.plot(
            layer_ints, [layers_data[layer]["twonn"][name]["dimension"] for layer in layers],
            linestyle="none", marker=marker, markersize=5, markerfacecolor="none",
            markeredgecolor=MUTED, label=f"random {name} of points",
        )
    ax_id.set_xticks(layer_ints)
    ax_id.set_ylim(bottom=0)
    ax_id.set_xlabel("layer", color=MUTED, fontsize=9)
    ax_id.set_ylabel("intrinsic dimension (TwoNN)", color=MUTED, fontsize=9)
    ax_id.set_title("Intrinsic dimension", color=INK, fontsize=10)
    ax_id.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    style(ax_id)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def plot_embeddings(inputs_path: str, embed_dir: str, out_path: Path) -> None:
    """Rows: t-SNE, UMAP (this layer's tuned setting), PCA; one column per layer."""
    inputs = np.load(inputs_path)
    layers = [int(layer) for layer in inputs["layers"]]
    rows = [("tsne2", "t-SNE"), ("umap2", "UMAP (tuned)"), ("pca2", "PCA")]
    fig, axes = plt.subplots(len(rows), len(layers), figsize=(2.6 * len(layers), 8.2), squeeze=False)
    for col, layer in enumerate(layers):
        embedded = np.load(Path(embed_dir) / f"layer{layer}.npz")
        best = json.loads((Path(embed_dir) / f"layer{layer}_scores.json").read_text())["umap_best"]
        colors = np.array(CLUSTER_COLORS)[inputs[f"cluster_{layer}"]]
        for row, (key, name) in enumerate(rows):
            ax = axes[row, col]
            xy = embedded[key]
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=0.6, alpha=0.6, linewidths=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if row == 0:
                ax.set_title(f"layer {layer}", color=INK, fontsize=10)
            if key == "umap2":
                ax.text(0.02, 0.02, f"n_neighbors {best['n_neighbors']}, min_dist {best['min_dist']}",
                        transform=ax.transAxes, color=MUTED, fontsize=7)
            if col == 0:
                ax.set_ylabel(name, color=INK, fontsize=10)
    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5, color=color, label=str(i))
        for i, color in enumerate(CLUSTER_COLORS)
    ]
    fig.legend(
        handles=handles, title="router cluster (K = 8)", loc="lower center", ncol=8,
        frameon=False, fontsize=8, title_fontsize=8, labelcolor=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=160)


if __name__ == "__main__":
    pca_path, inputs_path, embed_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    if Path(pca_path).exists():
        plot_variance_and_twonn(pca_path, out_dir / "jpp-pca-variance-twonn-2026-10-10.png")
    if Path(inputs_path).exists():
        plot_embeddings(inputs_path, embed_dir, out_dir / "jpp-tsne-umap-pca-2026-10-10.png")
