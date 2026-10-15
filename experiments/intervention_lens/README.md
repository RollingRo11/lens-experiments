# Intervention-trained lens (Gemma 4 E2B)

One 1536 x 1536 map per layer, learned by Adam on edit pairs instead of averaged from
gradients. Spec: `notes/gradient-trained-lens.md` in the project files. Code:
`src/workspace_lens/fitting/intervention_fitting.py` and the `collect-pairs`,
`fit-intervention`, `fit-output-kl` and `build-experts` stages of `scripts/jpp_cli.py`.

Jobs (run from `/projects/RohanKathuria/jpp_lens`, outputs in `artifacts/gemma4_e2b/intervention/`):

1. `slurm_dry_run.sh`: 16 prompts, all families, pair checks, short fits, evaluate.
2. `slurm_collect.sh` and `slurm_plain_jlens.sh` (independent): the pairs on 1,024 prompts,
   and the plain-gradient J-lens on 64 prompts (the reference for the gauss check).
3. `slurm_fit_and_score.sh`: runs 0-7, the map comparison (`compare_maps.csv`) and
   `evaluate` on the held-out items with the J++ (k-means 8), R-Lens and Logit baselines.

Gemma 4 E2B note: layers 15-34 reuse the keys and values of layers 13 and 14
(`num_kv_shared_layers` 20), so an edit at layer 16 or later cannot reach other positions.
At those layers the summed target equals the same-position target, for the J-lens too.
