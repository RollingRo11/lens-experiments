"""datamapplot 2D maps (each layer's tuned UMAP) for each clustering algorithm, plus the
cluster labels and colours the 3D page uses, so both views colour a cluster the same way.

Algorithms: the K = 8 router's k-means, ADP and HDBSCAN. The ADP / HDBSCAN setting for
each layer comes from cluster_layer.py's runs by a fixed rule (:func:`choose_settings`),
recorded in ``chosen.json``. Clusters are numbered by size (0 = largest). Each is named by
its 3 most common tokens; HDBSCAN noise is its own group, "noise", drawn grey.

    uv run --no-project --python 3.12 --with datamapplot \
        python experiments/dimensionality/build_maps.py
"""

from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter
from pathlib import Path

import datamapplot
import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NOISE_COLOR = "#999999"
NOISE_NAME = "noise"
ALGORITHM_NAMES = {"kmeans": "K-means (router, K = 8)", "adp": "ADP", "hdbscan": "HDBSCAN"}


def cluster_colors(num_clusters: int) -> list[str]:
    """The 8-colour palette when it is enough; otherwise hues spread by the golden angle
    with alternating lightness, so neighbouring cluster numbers differ."""
    if num_clusters <= len(PALETTE):
        return PALETTE[:num_clusters]
    colors = []
    for i in range(num_clusters):
        hue = (i * 0.381966) % 1.0
        lightness = 0.45 if i % 2 == 0 else 0.62
        r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.62)
        colors.append("#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255)))
    return colors


def token_text(label: str) -> str:
    token = label.split("⟦", 1)[1].rsplit("⟧", 1)[0]
    return token.strip().replace("\n", "↵") or "␣"


def choose_settings(runs: list[dict], min_clusters: int = 8) -> dict[str, str]:
    """ADP: the largest Z (fewest, most significant density peaks) that still leaves at
    least ``min_clusters`` clusters (the router's K), else the smallest Z tried.
    HDBSCAN: among settings with at least 3 clusters, the one calling the fewest points
    noise (in 64 dims most settings call most points noise or leave 1-2 giant clusters)."""
    adp = sorted((r for r in runs if r["algorithm"] == "adp"), key=lambda r: r["Z"])
    adp_ok = [r for r in adp if r["num_clusters"] >= min_clusters]
    adp_pick = adp_ok[-1] if adp_ok else adp[0]
    hdbscan = [r for r in runs if r["algorithm"] == "hdbscan"]
    hdbscan_ok = [r for r in hdbscan if r["num_clusters"] >= 3] or hdbscan
    hdbscan_pick = min(hdbscan_ok, key=lambda r: r["noise_fraction"])
    return {
        "adp": f"adp_Z{adp_pick['Z']}",
        "hdbscan": f"hdbscan_m{hdbscan_pick['min_cluster_size']}_{hdbscan_pick['selection']}",
    }


def readable_setting(key: str) -> str:
    """``adp_Z3.0`` -> ``Z = 3.0``; ``hdbscan_m100_eom`` -> ``min_cluster_size 100, eom``."""
    if key.startswith("adp_Z"):
        return f"Z = {key.removeprefix('adp_Z')}"
    _, min_size, selection = key.split("_")
    return f"min_cluster_size {min_size.removeprefix('m')}, {selection}"


def rank_by_size(labels_S: np.ndarray) -> np.ndarray:
    """Renumber clusters 0, 1, ... by decreasing size; noise (< 0) stays -1."""
    clusters, counts = np.unique(labels_S[labels_S >= 0], return_counts=True)
    order = clusters[np.argsort(-counts, kind="stable")]
    rank = {cluster: i for i, cluster in enumerate(order)}
    return np.array([rank.get(label, -1) for label in labels_S])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="results/dimensionality/embed_inputs.npz")
    parser.add_argument("--embed-dir", default="results/dimensionality/embed_3d")
    parser.add_argument("--clusters-dir", default="results/dimensionality/clusters")
    parser.add_argument("--out-dir", default="results/dimensionality/maps")
    args = parser.parse_args()

    inputs = np.load(args.inputs)
    layers = [int(layer) for layer in inputs["layers"]]
    chosen = {
        str(layer): choose_settings(json.loads(
            (Path(args.clusters_dir) / f"layer{layer}_runs.json").read_text())["runs"])
        for layer in layers
    }
    (Path(args.clusters_dir) / "chosen.json").write_text(json.dumps(chosen, indent=1))
    tokens = [token_text(label) for label in inputs["label"].tolist()]
    contexts = [label.replace("\n", "↵") for label in inputs["label"].tolist()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_data: dict = {"algorithms": ALGORITHM_NAMES, "layers": {}}
    for layer in layers:
        clusterings = np.load(Path(args.clusters_dir) / f"layer{layer}.npz")
        layer_choice = chosen[str(layer)]
        raw = {
            "kmeans": inputs[f"cluster_{layer}"].astype(np.int64),
            "adp": clusterings[layer_choice["adp"]],
            "hdbscan": clusterings[layer_choice["hdbscan"]],
        }
        ranked = {algorithm: rank_by_size(labels_S) for algorithm, labels_S in raw.items()}
        hover = [
            f"{contexts[i]}  ·  position {inputs['position'][i]}  ·  "
            + "  ·  ".join(
                f"{name} {'noise' if ranked[a][i] < 0 else ranked[a][i]}"
                for a, name in (("kmeans", "K-means"), ("adp", "ADP"), ("hdbscan", "HDBSCAN"))
            )
            for i in range(len(tokens))
        ]
        umap_2d = np.load(Path(args.embed_dir) / f"layer{layer}.npz")["umap2"]

        page_data["layers"][str(layer)] = {}
        for algorithm, labels_S in ranked.items():
            num_clusters = int(labels_S.max()) + 1
            colors = cluster_colors(num_clusters)
            names = []
            for cluster in range(num_clusters):
                top = Counter(tokens[i] for i in np.flatnonzero(labels_S == cluster)).most_common(3)
                names.append(f"#{cluster} " + " · ".join(token for token, _ in top))
            point_names = np.array([NOISE_NAME if c < 0 else names[c] for c in labels_S])
            noise_fraction = float((labels_S < 0).mean())
            setting = {"kmeans": "fit on all 110,804 points",
                       "adp": readable_setting(layer_choice["adp"]),
                       "hdbscan": readable_setting(layer_choice["hdbscan"])}[algorithm]

            plot = datamapplot.create_interactive_plot(
                umap_2d,
                point_names,
                hover_text=hover,
                noise_label=NOISE_NAME,
                noise_color=NOISE_COLOR,
                label_color_map={**dict(zip(names, colors)), NOISE_NAME: NOISE_COLOR},
                enable_search=True,
                title=f"Layer {layer} · {ALGORITHM_NAMES[algorithm]}",
                sub_title=(f"{num_clusters} clusters"
                           + (f", {noise_fraction:.0%} noise" if noise_fraction else "")
                           + f" · {setting} · tuned UMAP"),
                title_font_size=22,
                sub_title_font_size=13,
                min_fontsize=12,
                max_fontsize=22,
            )
            plot.save(str(out_dir / f"layer{layer}_{algorithm}.html"))
            page_data["layers"][str(layer)][algorithm] = {
                "labels": labels_S.tolist(),
                "colors": colors,
                "names": names,
                "setting": setting,
                "noise_fraction": noise_fraction,
            }
            print(f"layer {layer} {algorithm}: {num_clusters} clusters, noise {noise_fraction:.2f}",
                  flush=True)

    (out_dir / "clusterings.json").write_text(json.dumps(page_data))


if __name__ == "__main__":
    main()
