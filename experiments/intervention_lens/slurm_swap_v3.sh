#!/bin/bash
# The biggest swap dataset (RAVEL + LRE with all templates and up to 1,000 words per relation,
# plus CounterFact, PARAREL, MUSE and WordNet), an all-layer lens trained on it from the identity
# with batches of 256 (half the steps at all layers, half at a random window of 1/3/5 layers),
# then: swap success on the held-out relations per single layer / window / all / the middle band
# 8-26, probe-swap over the middle band, and exact (fp32, uncapped) recall@10.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 08:00:00
#SBATCH -J ilens-swapv3
#SBATCH -o logs/intervention_swap_v3_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
D=$I/swap_data_v3
X=data_ext
CSV=$OUT/model_correctness_stripped.csv
STEPS=${STEPS:-4000}
ALL=$(seq -s, 0 33)
BAND=$(seq -s, 8 26)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== build dataset $(date +%T)"
[ -f $D/train.json ] || $PY experiments/intervention_lens/build_swap_dataset_v2.py --ravel-dir $X/ravel \
    --lre-dir $X/relations/data --extra-relations $X/relations_extra/*.json \
    --templates-per-relation 20 --subjects-per-relation 1000 --pairs-per-template 300 \
    --max-heldout-items 1500 --out-dir $D 2>&1 | grep --line-buffered -v httpx

echo "== fit $(date +%T)"
[ -f $I/swapall_v3.pt ] || $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $ALL \
    --mode supervised --swap-data $D/train.json --init identity --steps $STEPS --batch-size 256 \
    --eval-every 250 --patience 1000 --heldout-examples 1024 --monitor-data $D/heldout.json \
    --all-layers-probability 0.5 --window-sizes 1,3,5 \
    --wandb-project lens-training --checkpoint-name swapall_v3 --out $I/swapall_v3.pt 2>&1 | grep --line-buffered -v -e httpx -e "dropping swap item"

ALL_LAYER_LENSES="$I/swapall_v3.pt $I/swapall_identity.pt $I/plain_jlens64_all.pt"
echo "== swap success per layer set, held-out relations $(date +%T)"
$PY experiments/intervention_lens/eval_layer_windows.py --data $D/heldout.json --band 8-26 --max-items 1500 \
    --lens $ALL_LAYER_LENSES logit:$ALL $I/swapdata2_identity.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt \
    --out $I/swapv3_windows_heldout.csv 2>&1 | grep --line-buffered -v -e httpx -e "dropping swap item"
echo "== probe-swap over the middle band $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $BAND --lens $ALL_LAYER_LENSES logit \
    --out $I/swapv3_probe_swap_band.parquet 2>&1 | grep --line-buffered -v -e httpx -e "ran trial"
echo "== probe-swap at the 7 layers $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS \
    --lens $ALL_LAYER_LENSES $I/swapdata2_identity.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt logit \
    --out $I/swapv3_probe_swap_7.parquet 2>&1 | grep --line-buffered -v -e httpx -e "ran trial"
echo "== exact recall@10 at the 7 layers $(date +%T)"
$PY experiments/intervention_lens/eval_tie_check.py --mode exact --correctness-csv $CSV \
    --out $I/eval_held_out_swapv3.csv --lens $ALL_LAYER_LENSES $I/swapdata2_identity.pt \
    $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt $I/plain_jlens64.pt logit 2>&1 | grep --line-buffered -v -e httpx -e "scored item" -e "recorded item"
echo "SWAP_V3_DONE $(date +%T)"
