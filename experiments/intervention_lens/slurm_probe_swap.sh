#!/bin/bash
# Probe-swap (the causal swap eval) on the baselines and the intervention-trained lenses: the
# "swapping performance" any new loss has to beat.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J ilens-swap
#SBATCH -o logs/intervention_probe_swap_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS \
    --lens $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt $I/plain_jlens64.pt logit \
    $I/ilens_gauss.pt $I/ilens_swap_a1_zero.pt $I/ilens_mix_zero.pt \
    --out $I/probe_swap.parquet 2>&1 | grep --line-buffered -v httpx
echo PROBE_SWAP_DONE
