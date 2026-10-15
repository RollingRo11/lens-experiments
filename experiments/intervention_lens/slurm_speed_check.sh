#!/bin/bash
# A short fit-swap run with the v3 settings and the speed-ups on (length-grouped batches; the
# stored clean pass when it fits next to training), for the real seconds per step with
# evaluations, to compare with the v3 run (same settings, before the speed-ups).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J ilens-speed
#SBATCH -o logs/intervention_speed_check_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
D=$I/swap_data_v3
STEPS=${STEPS:-500}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p $I/speed_check
echo "== fit-swap, $STEPS steps $(date +%T)"
$PY -u scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $(seq -s, 0 33) \
    --mode supervised --swap-data $D/train.json --init identity --steps $STEPS --batch-size 256 \
    --eval-every 250 --patience 1000 --heldout-examples 1024 --monitor-data $D/heldout.json \
    --all-layers-probability 0.5 --window-sizes 1,3,5 \
    --checkpoint-name speed_check --out $I/speed_check/speed_check.pt \
    2>&1 | grep --line-buffered -v -e httpx -e "dropping swap item"
echo "SPEED_CHECK_DONE $(date +%T)"
