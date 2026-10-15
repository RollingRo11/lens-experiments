# k-means loss vs K on the router's activations

- `extract_and_sweep.py`: re-records the activations `fit-router` clustered, using the settings in `router_stamp.json`. It saves them to `artifacts/gemma4_e2b/activations/`, checks them against `router.pt`, and runs k-means on the router's 64-dim projection for each K and seed. Results go to `results/kmeans_k_sweep/sweep.json`.
- `slurm_kmeans_k_sweep.sh`: runs the script as one H200 job from the cluster repo root (about 2 min).
- `plot_sweep.py`: one panel per layer of loss vs K: `uv run --with matplotlib python experiments/kmeans_k_sweep/plot_sweep.py results/kmeans_k_sweep/sweep.json out.png`

Plot and notes: `~/Developer/researchnotes/jpp-lens.md` (10/10/2026).
