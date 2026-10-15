# Dimensionality of the router's activations

- `pca_and_twonn.py` (+ `slurm_pca_and_twonn.sh`, A100): the PCA variance spectrum and TwoNN intrinsic dimension of the saved activations (`artifacts/gemma4_e2b/activations/`, unit-normalised, full d_model). Writes `results/dimensionality/pca_and_twonn.json`.
- `prep_embed_inputs.py` (+ `slurm_embed_prep.sh`, CPU): picks 20k rows and writes their 64-dim points, router clusters and hover labels to `results/dimensionality/embed_inputs.npz`.
- `embed_layer.py` (+ `slurm_embed_layers.sh`, CPU array with one task per layer, own uv env): PCA, t-SNE and UMAP in 2D and 3D. UMAP is tuned per layer on trustworthiness and distance correlation. Writes `results/dimensionality/embed_3d/layer{L}.npz` and `layer{L}_scores.json`.
- `cluster_layer.py` (+ `slurm_cluster_layers.sh`, CPU array, own uv env with numpy<2 so DADApy builds, see `build-constraints.txt`): ADP (DADApy) and HDBSCAN on each layer's 20k shown points in 64 dims, for a few settings each. Writes `results/dimensionality/clusters/`.
- `build_maps.py` (+ `slurm_build_maps.sh`, CPU): picks one ADP and one HDBSCAN setting per layer (`choose_settings`), then writes a datamapplot map per layer × clustering (K-means, ADP, HDBSCAN) and `clusterings.json` (labels, names, colours) to `results/dimensionality/maps/`.
- `plot_dimensionality.py`: the variance/TwoNN figure and the 3-row (t-SNE, UMAP, PCA) figure. Needs `uv run --with matplotlib`.
- `build_3d_html.py`: the interactive 3D page (plotly.js, data embedded) with the clustering picker. It frames the datamapplot maps from a folder next to it.
- `embed_2d.py`: the first, untuned 2D run (laptop). Superseded by `embed_layer.py`.

Plots and notes: `~/Developer/researchnotes/jpp-lens.md` (10/10/2026).
