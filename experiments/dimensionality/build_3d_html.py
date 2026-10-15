"""One HTML page with the 3D PCA, t-SNE and tuned UMAP of each layer (plotly.js), coloured
by a clustering you pick (the K = 8 router's k-means, ADP or HDBSCAN), with each point's
token and the 10 tokens before it on hover. Below the 3D views, the matching 2D
datamapplot map (from build_maps.py) loads in a frame.

The coordinates are embedded as base64 float32; plotly.js loads from jsDelivr and the
datamapplot maps are read from ``MAPS_HREF`` (a folder next to the page).

    python experiments/dimensionality/build_3d_html.py results/dimensionality/embed_inputs.npz \
        results/dimensionality/embed_3d results/dimensionality/maps/clusterings.json MAPS_HREF OUT_HTML
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import numpy as np

PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def b64_float32(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.float32).tobytes()).decode()


def build_data(inputs_path: str, embed_dir: str, clusterings_path: str, maps_href: str) -> dict:
    inputs = np.load(inputs_path)
    clusterings = json.loads(Path(clusterings_path).read_text())
    labels = [
        html.escape(label).replace("\n", "↵") for label in inputs["label"].tolist()
    ]
    data = {
        "num_points": len(labels),
        "labels": labels,
        "position": inputs["position"].tolist(),
        "prompt": inputs["prompt_idx"].tolist(),
        "algorithms": clusterings["algorithms"],
        "maps_href": maps_href,
        "layers": [],
    }
    for layer in [int(layer) for layer in inputs["layers"]]:
        embedded = np.load(Path(embed_dir) / f"layer{layer}.npz")
        scores = json.loads((Path(embed_dir) / f"layer{layer}_scores.json").read_text())
        layer_clusterings = clusterings["layers"][str(layer)]
        for clustering in layer_clusterings.values():
            clustering["names"] = [html.escape(name) for name in clustering["names"]]
        data["layers"].append(
            {
                "layer": layer,
                "clusterings": layer_clusterings,
                "pca": b64_float32(embedded["pca3"]),
                "tsne": b64_float32(embedded["tsne3"]),
                "umap": b64_float32(embedded["umap3"]),
                "umap_best": scores["umap_best"],
                "scores": scores["table"],
            }
        )
    return data


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Router Activations 3D</title>
<script src="__PLOTLY__"></script>
<style>
  :root {
    color-scheme: light;
    --surface-0: #f5f4f1; --surface-1: #fcfcfb; --border: #e4e3df;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
    --accent: #2a78d6; --grid: #e4e3df;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
      --accent: #3987e5; --grid: #383835;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --accent: #3987e5; --grid: #383835;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--surface-0); color: var(--text-primary);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1400px; margin: 0 auto; padding: 20px 16px 32px; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--text-secondary); margin: 0 0 16px; font-size: 13px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; margin-bottom: 10px; }
  .label { color: var(--text-muted); font-size: 12px; margin-right: 6px; }
  .seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; vertical-align: middle; }
  .seg button { border: 0; background: var(--surface-1); color: var(--text-secondary);
                padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer; }
  .seg button + button { border-left: 1px solid var(--border); }
  .seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
  .clusters { margin: 0 0 12px; }
  .clusters-head { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; margin-bottom: 6px; }
  .clusters-head .setting { color: var(--text-muted); font-size: 12px; }
  .linkbtn { border: 0; background: none; color: var(--accent); font: inherit; font-size: 12px; cursor: pointer; padding: 0; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; max-height: 92px; overflow-y: auto; }
  .chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--border);
          background: var(--surface-1); color: var(--text-secondary); border-radius: 999px;
          padding: 3px 10px; font: inherit; font-size: 12px; cursor: pointer; max-width: 260px; }
  .chip span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .chip i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
  .chip[aria-pressed="false"] { opacity: 0.4; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 10px 10px 4px; min-width: 0; }
  .card h2 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
  .card .hint { color: var(--text-muted); font-size: 12px; margin: 0; }
  .plot { height: 440px; }
  .map-card { margin-top: 12px; padding-bottom: 10px; }
  .map-card iframe { width: 100%; height: 640px; border: 1px solid var(--border); border-radius: 8px; margin-top: 8px; background: #fff; }
  .map-card a { color: var(--accent); font-size: 12px; }
  table { border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  th, td { padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--text-muted); font-weight: 500; }
  .best { color: var(--accent); font-weight: 600; }
  .tables { display: flex; flex-wrap: wrap; gap: 32px; margin-top: 16px; }
  .tables h3 { font-size: 13px; font-weight: 600; margin: 0; }
  .note { color: var(--text-muted); font-size: 12px; margin: 4px 0 0; max-width: 520px; }
  .scroll { overflow-x: auto; }
</style>
</head>
<body>
<main>
  <h1>Router activations in 3D: Gemma 4 E2B</h1>
  <p class="sub">__N__ of 110,804 points per layer, in the router's 64-dim PCA projection.
    Colour = the point's cluster under the clustering you pick. Clusters are numbered by size
    (#0 = largest) and named by their 3 most common tokens. Drag to rotate, scroll to zoom,
    hover for the token ⟦like this⟧ and the 10 tokens before it.</p>
  <div class="controls">
    <div><span class="label">Layer</span><span class="seg" id="layers"></span></div>
    <div><span class="label">Clustering</span><span class="seg" id="algorithms"></span></div>
  </div>
  <div class="clusters">
    <div class="clusters-head">
      <span class="label" id="cluster-count"></span>
      <span class="setting" id="cluster-setting"></span>
      <button class="linkbtn" id="show-all">show all</button>
      <button class="linkbtn" id="hide-all">hide all</button>
    </div>
    <div class="chips" id="chips"></div>
  </div>
  <div class="grid">
    <div class="card"><h2>PCA</h2><p class="hint">top 3 principal directions</p><div class="plot" id="plot-pca"></div></div>
    <div class="card"><h2>t-SNE</h2><p class="hint">perplexity 30, PCA start</p><div class="plot" id="plot-tsne"></div></div>
    <div class="card"><h2>UMAP (tuned for this layer)</h2><p class="hint" id="umap-hint"></p><div class="plot" id="plot-umap"></div></div>
  </div>
  <div class="card map-card">
    <h2>2D map (datamapplot)</h2>
    <p class="hint">The same layer, clustering and colours, on the tuned 2D UMAP. Labels are cluster
      names; zoom for more. The search box finds tokens. <a id="map-link" target="_blank">Open on its own page</a></p>
    <iframe id="map" loading="lazy" title="datamapplot map"></iframe>
  </div>
  <div class="tables">
    <div><h3>How faithful is each 3D view?</h3><div class="scroll"><table id="scores"></table></div>
      <p class="note">Trustworthiness (k = 15): are a point's neighbours in the view also its neighbours in 64 dims? Distance correlation: Spearman correlation of pairwise distances, 64 dims vs the view. 1 = perfect for both.</p></div>
    <div><h3>UMAP settings tried (2D, mean of the two scores)</h3><div class="scroll"><table id="grid"></table></div>
      <p class="note">The setting with the highest mean is used for the 2D views and for the 3D UMAP.</p></div>
  </div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const NOISE_COLOR = "#999999";
const METHODS = ["pca", "tsne", "umap"];
const ALGORITHMS = Object.keys(DATA.algorithms);
let current = 0, algorithm = ALGORITHMS[0];
const hidden = new Set();
const coords = {}, groupsCache = {};

const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function decode(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
const hoverText = DATA.labels.map((label, i) =>
  label + "<br>position " + DATA.position[i] + " · prompt " + DATA.prompt[i]);

function clustering() { return DATA.layers[current].clusterings[algorithm]; }

function groupsFor(method) {
  // Points grouped by cluster (noise = -1 last) for this layer, method and clustering.
  const key = current + ":" + method + ":" + algorithm;
  if (!groupsCache[key]) {
    const ckey = current + ":" + method;
    if (!coords[ckey]) coords[ckey] = decode(DATA.layers[current][method]);
    const xyz = coords[ckey], c = clustering();
    const groups = new Map();
    c.labels.forEach((label, i) => {
      if (!groups.has(label)) groups.set(label, {x: [], y: [], z: [], text: []});
      const g = groups.get(label);
      g.x.push(xyz[3*i]); g.y.push(xyz[3*i+1]); g.z.push(xyz[3*i+2]);
      g.text.push(hoverText[i] + "<br>" + (label < 0 ? "noise" : c.names[label]));
    });
    groupsCache[key] = [...groups.entries()].sort((a, b) => (a[0] < 0) - (b[0] < 0) || a[0] - b[0]);
  }
  return groupsCache[key];
}

function tracesFor(method) {
  const c = clustering();
  return groupsFor(method).map(([label, g]) => ({
    type: "scatter3d", mode: "markers", name: label < 0 ? "noise" : "#" + label,
    x: g.x, y: g.y, z: g.z, text: g.text, hovertemplate: "%{text}<extra></extra>",
    visible: !hidden.has(label),
    marker: {size: label < 0 ? 1.5 : 2, color: label < 0 ? NOISE_COLOR : c.colors[label],
             opacity: label < 0 ? 0.35 : 0.75},
  }));
}

function layoutFor() {
  const axis = {showticklabels: false, title: "", showbackground: false,
                gridcolor: css("--grid"), zerolinecolor: css("--grid"), showspikes: false};
  return {
    paper_bgcolor: "rgba(0,0,0,0)", margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false,
    uirevision: "layer" + current,
    font: {color: css("--text-secondary"), size: 11},
    hoverlabel: {bgcolor: css("--surface-1"), bordercolor: css("--border"),
                 font: {color: css("--text-primary"), size: 12}},
    scene: {xaxis: axis, yaxis: axis, zaxis: axis, aspectmode: "cube"},
  };
}

function renderPlots() {
  for (const method of METHODS) {
    Plotly.react("plot-" + method, tracesFor(method), layoutFor(), {responsive: true, displaylogo: false});
  }
}

function renderChips() {
  const c = clustering(), chips = document.getElementById("chips");
  const noise = c.labels.some(l => l < 0);
  const n = c.names.length;
  document.getElementById("cluster-count").textContent =
    n + " clusters" + (noise ? " + noise (" + Math.round(100 * c.noise_fraction) + "%)" : "");
  document.getElementById("cluster-setting").textContent = c.setting;
  chips.innerHTML = "";
  const entries = c.names.map((name, label) => [label, name, c.colors[label]]);
  if (noise) entries.push([-1, "noise", NOISE_COLOR]);
  for (const [label, name, color] of entries) {
    const b = document.createElement("button");
    b.className = "chip"; b.title = name;
    b.setAttribute("aria-pressed", String(!hidden.has(label)));
    b.innerHTML = '<i style="background:' + color + '"></i><span>' + name + "</span>";
    b.addEventListener("click", () => {
      hidden.has(label) ? hidden.delete(label) : hidden.add(label);
      b.setAttribute("aria-pressed", String(!hidden.has(label)));
      renderPlots();
    });
    chips.appendChild(b);
  }
}

function setAll(show) {
  hidden.clear();
  if (!show) {
    clustering().names.forEach((_, label) => hidden.add(label));
    hidden.add(-1);
  }
  renderChips(); renderPlots();
}

function renderMap() {
  const src = DATA.maps_href + "/layer" + DATA.layers[current].layer + "_" + algorithm + ".html";
  const frame = document.getElementById("map");
  if (frame.getAttribute("src") !== src) frame.setAttribute("src", src);
  document.getElementById("map-link").setAttribute("href", src);
}

function render() {
  const layer = DATA.layers[current];
  document.getElementById("umap-hint").textContent =
    "n_neighbors " + layer.umap_best.n_neighbors + ", min_dist " + layer.umap_best.min_dist;
  document.querySelectorAll("#layers button").forEach((b, i) =>
    b.setAttribute("aria-pressed", String(i === current)));
  document.querySelectorAll("#algorithms button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.algorithm === algorithm)));
  renderChips(); renderPlots(); renderTables(layer); renderMap();
}

function renderTables(layer) {
  const fmt = v => v.toFixed(3);
  const rows3d = layer.scores.filter(r => r.dims === 3);
  const name = {pca: "PCA", tsne: "t-SNE", umap: "UMAP (tuned)"};
  document.getElementById("scores").innerHTML =
    "<tr><th>view</th><th>trustworthiness</th><th>distance corr.</th></tr>" +
    rows3d.map(r => "<tr><td>" + name[r.method] + "</td><td>" + fmt(r.trust) +
                    "</td><td>" + fmt(r.dist_corr) + "</td></tr>").join("");
  const grid = layer.scores.filter(r => r.method === "umap" && r.dims === 2);
  const neighbors = [...new Set(grid.map(r => r.n_neighbors))];
  const dists = [...new Set(grid.map(r => r.min_dist))];
  const best = layer.umap_best;
  document.getElementById("grid").innerHTML =
    "<tr><th>n_neighbors</th>" + dists.map(d => "<th>min_dist " + d + "</th>").join("") + "</tr>" +
    neighbors.map(n => "<tr><td>" + n + "</td>" + dists.map(d => {
      const r = grid.find(r => r.n_neighbors === n && r.min_dist === d);
      const isBest = n === best.n_neighbors && d === best.min_dist;
      return "<td" + (isBest ? ' class="best"' : "") + ">" + fmt(r.mean) + "</td>";
    }).join("") + "</tr>").join("");
}

function buildControls() {
  const layers = document.getElementById("layers");
  DATA.layers.forEach((layer, i) => {
    const b = document.createElement("button");
    b.textContent = layer.layer;
    b.addEventListener("click", () => { current = i; hidden.clear(); render(); });
    layers.appendChild(b);
  });
  const algorithms = document.getElementById("algorithms");
  for (const key of ALGORITHMS) {
    const b = document.createElement("button");
    b.textContent = DATA.algorithms[key]; b.dataset.algorithm = key;
    b.addEventListener("click", () => { algorithm = key; hidden.clear(); render(); });
    algorithms.appendChild(b);
  }
  document.getElementById("show-all").addEventListener("click", () => setAll(true));
  document.getElementById("hide-all").addEventListener("click", () => setAll(false));
}

buildControls();
render();
darkQuery.addEventListener("change", renderPlots);
</script>
</body>
</html>
"""


def main(inputs_path: str, embed_dir: str, clusterings_path: str, maps_href: str, out_path: str) -> None:
    data = build_data(inputs_path, embed_dir, clusterings_path, maps_href)
    page = (
        PAGE.replace("__PLOTLY__", PLOTLY_JS)
        .replace("__N__", f"{data['num_points']:,}")
        .replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    )
    Path(out_path).write_text(page)
    print(f"wrote {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:6])
