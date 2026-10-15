#!/bin/bash
# The swap-trained lens with one matrix for every layer (0-33), from the identity, on the
# RAVEL + LRE + hand-written swap dataset. Half the steps swap at every layer; the other half
# at one random window of 1, 3 or 5 consecutive layers, the rest left alone. Then: swap success
# on the held-out relations per single layer / window of 3 / window of 5 / all, and recall@10
# per layer set, next to the baselines (whose maps exist only at the 7 band layers).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J ilens-swapall
#SBATCH -o logs/intervention_swap_all_layers_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
D=$I/swap_data_v2
X=data_ext
CSV=$OUT/model_correctness_stripped.csv
STEPS=${STEPS:-6000}
ALL=$(seq -s, 0 33)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f $D/train.json ] || $PY experiments/intervention_lens/build_swap_dataset_v2.py --ravel-dir $X/ravel \
    --lre-dir $X/relations/data --max-heldout-items 600 --out-dir $D 2>&1 | grep --line-buffered -v httpx

echo "== fit $(date +%T)"
[ -f $I/swapall_identity.pt ] || $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $ALL \
    --mode supervised --swap-data $D/train.json --init identity --steps $STEPS --eval-every 250 \
    --patience 1000 --heldout-examples 512 --monitor-data $D/heldout.json \
    --all-layers-probability 0.5 --window-sizes 1,3,5 \
    --checkpoint-name swapall_identity --out $I/swapall_identity.pt 2>&1 | grep --line-buffered -v httpx

echo "== swap success per layer set $(date +%T)"
$PY experiments/intervention_lens/eval_layer_windows.py --data $D/heldout.json \
    --lens $I/swapall_identity.pt logit:$ALL $I/swapdata2_identity.pt $I/plain_jlens64.pt $OUT/jpp_lens_v2.pt \
    --out $I/swapall_windows_heldout.csv 2>&1 | grep --line-buffered -v httpx

echo "== recall@10, all layers $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $ALL --correctness-csv $CSV \
    --lens $I/swapall_identity.pt logit --out $I/eval_held_out_swapall_layers.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
$PY experiments/intervention_lens/recall_by_layer.py --ranks $I/eval_held_out_swapall_layers.csv \
    --out $I/swapall_recall_by_layer.csv
echo "== recall@10, band layers $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $I/swapall_identity.pt $I/swapdata2_identity.pt $I/plain_jlens64.pt $OUT/jpp_lens_v2.pt \
    $OUT/pooled_lens.pt logit --out $I/eval_held_out_swapall_band.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
echo "SWAP_ALL_LAYERS_DONE $(date +%T)"
