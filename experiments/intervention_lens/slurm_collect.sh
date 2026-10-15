#!/bin/bash
# Collect the clean pool and the edit pairs of the intervention-trained lens on the 1,024
# WikiText prompts of the J++ fit. FAMILIES defaults to all four families of runs 1-3; a
# later call on the same output directory reuses the clean pool, so the families can be
# split over jobs (the per-prompt draws depend only on the seed and the prompt, so the swap
# families stay paired across jobs).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J ilens-collect
#SBATCH -o logs/intervention_collect_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
FAMILIES=${FAMILIES:-gauss:0.05,swap:1,swap:0.5,swap:0.25}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== collect-pairs $FAMILIES $(date +%T)"
$PY scripts/jpp_cli.py collect-pairs --hf-model-name $MODEL --layers $LAYERS --num-prompts 1024 \
    --families $FAMILIES --pairs-per-prompt 32 --rows-per-pass 128 \
    --out-dir $OUT/intervention 2>&1 | grep --line-buffered -v httpx
du -sh $OUT/intervention/*
echo "COLLECT_DONE $(date +%T)"
