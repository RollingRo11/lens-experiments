#!/bin/bash
# The v3 all-layer lens trained longer (STEPS, default 16,000 = 4x the v3 run; the cosine decay
# stretches over all of them), with the faster trainer (length-grouped batches; the stored clean
# pass when it fits). Same data and settings as slurm_swap_v3.sh otherwise, logged live to W&B.
# Then the v3 evals for this lens alone (the v3 job scored the others on the same data):
# swap success per layer set on the held-out relations, probe-swap over the band and at the
# 7 layers, exact recall@10.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -J ilens-swapv3long
#SBATCH -o logs/intervention_swap_v3_long_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
D=$I/swap_data_v3
CSV=$OUT/model_correctness_stripped.csv
STEPS=${STEPS:-16000}
NAME=swapall_v3_${STEPS}
ALL=$(seq -s, 0 33)
BAND=$(seq -s, 8 26)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
FILTER=(grep --line-buffered -v -e httpx -e "dropping swap item" -e "ran trial" -e "scored item" -e "recorded item")

echo "== fit $STEPS steps $(date +%T)"
[ -f $I/$NAME.pt ] || $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $ALL \
    --mode supervised --swap-data $D/train.json --init identity --steps $STEPS --batch-size 256 \
    --eval-every 500 --patience 1000 --heldout-examples 1024 --monitor-data $D/heldout.json \
    --all-layers-probability 0.5 --window-sizes 1,3,5 \
    --wandb-project lens-training --checkpoint-name $NAME --out $I/$NAME.pt 2>&1 | "${FILTER[@]}"

echo "== swap success per layer set, held-out relations $(date +%T)"
$PY experiments/intervention_lens/eval_layer_windows.py --data $D/heldout.json --band 8-26 --max-items 1500 \
    --lens $I/$NAME.pt --out $I/${NAME}_windows_heldout.csv 2>&1 | "${FILTER[@]}"
echo "== probe-swap over the middle band $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $BAND --lens $I/$NAME.pt \
    --out $I/${NAME}_probe_swap_band.parquet 2>&1 | "${FILTER[@]}"
echo "== probe-swap at the 7 layers $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS --lens $I/$NAME.pt \
    --out $I/${NAME}_probe_swap_7.parquet 2>&1 | "${FILTER[@]}"
echo "== exact recall@10 at the 7 layers $(date +%T)"
$PY experiments/intervention_lens/eval_tie_check.py --mode exact --correctness-csv $CSV \
    --out $I/eval_held_out_${NAME}.csv --lens $I/$NAME.pt 2>&1 | "${FILTER[@]}"
echo "SWAP_V3_LONG_DONE $(date +%T)"
