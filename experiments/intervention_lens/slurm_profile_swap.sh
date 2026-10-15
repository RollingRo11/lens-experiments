#!/bin/bash
# Times each part of a swap-training step (profile_swap_step.py) on the v3 swap dataset, then a
# short fit-swap run with the speed-ups on (length-grouped batches, stored clean pass) in fp32
# and bf16 edits, for the real seconds per step with evaluations.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J ilens-profile
#SBATCH -o logs/intervention_profile_swap_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
D=$I/swap_data_v3
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u experiments/intervention_lens/profile_swap_step.py --swap-data $D/train.json \
    2>&1 | grep --line-buffered -v -e httpx -e "dropping swap item"
mkdir -p $I/speed_check
for PRECISION in fp32 model; do
    echo "== fit-swap, 300 steps, edit precision $PRECISION $(date +%T)"
    $PY -u scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $(seq -s, 0 33) \
        --mode supervised --swap-data $D/train.json --init identity --steps 300 --batch-size 256 \
        --eval-every 100 --patience 1000 --heldout-examples 1024 --monitor-data $D/heldout.json \
        --all-layers-probability 0.5 --window-sizes 1,3,5 --edit-precision $PRECISION \
        --checkpoint-name speed_$PRECISION --out $I/speed_check/speed_$PRECISION.pt \
        2>&1 | grep --line-buffered -v -e httpx -e "dropping swap item"
done
echo "PROFILE_DONE $(date +%T)"
