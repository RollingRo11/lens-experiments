#!/bin/bash
# Dry run of the swap-loss trainer: both modes, 64 prompts, 40 steps from the plain J-lens,
# to time a step and check the loss moves before the full runs.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J ilens-swapdry
#SBATCH -o logs/intervention_swap_dry_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
mkdir -p $I/swap_dry
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for MODE in next_token counterfactual; do
    echo "== $MODE $(date +%T)"
    $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $LAYERS --num-prompts 64 \
        --mode $MODE --init $I/plain_jlens64.pt --steps 40 --eval-every 20 --heldout-examples 64 \
        --holdout-fraction 0.25 --checkpoint-name swap_dry_$MODE --out $I/swap_dry/$MODE.pt \
        2>&1 | grep --line-buffered -v httpx
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo "SWAP_DRY_DONE $(date +%T)"
