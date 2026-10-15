#!/bin/bash
# Control for the removal-edit mix: the same fit-weights over the Gaussian map and the R-Lens
# only (no swap maps), scored next to the full mix.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J ilens-mixctl
#SBATCH -o logs/intervention_mix_control_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
CSV=$OUT/model_correctness_stripped.csv
$PY scripts/jpp_cli.py build-experts --layers $LAYERS --lens $I/ilens_gauss.pt $OUT/pooled_lens.pt \
    --out $I/experts_ilens_mix_gauss_rlens.pt
$PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --experts $I/experts_ilens_mix_gauss_rlens.pt --checkpoint-name ilens_mix_gauss_rlens --top-n 2 \
    --out $I/ilens_mix_gauss_rlens.pt 2>&1 | grep --line-buffered -v -e httpx -e "recorded item"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $I/ilens_mix_gauss_rlens.pt $I/ilens_mix_zero.pt $OUT/jpp_lens_v2.pt \
    --out $I/eval_held_out_mix_control.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
echo MIX_CONTROL_DONE
