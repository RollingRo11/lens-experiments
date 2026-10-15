#!/bin/bash
# Build the templated swap dataset, train the band maps on it (supervised swap loss) from
# the plain J-lens and from the identity, then score: swap success on the held-out domains,
# probe-swap, and recall@10, next to the baselines.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J ilens-swapdata
#SBATCH -o logs/intervention_swap_dataset_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
D=$I/swap_data
CSV=$OUT/model_correctness_stripped.csv
STEPS=${STEPS:-2000}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== build dataset $(date +%T)"
[ -f $D/train.json ] || $PY experiments/intervention_lens/build_swap_dataset.py --out-dir $D 2>&1 | grep --line-buffered -v httpx

echo "== fits $(date +%T)"
for INIT in jlens identity; do
    INIT_PATH=$([ $INIT = jlens ] && echo $I/plain_jlens64.pt || echo identity)
    [ -f $I/swapdata_$INIT.pt ] || $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $LAYERS \
        --mode supervised --swap-data $D/train.json --init $INIT_PATH --steps $STEPS \
        --checkpoint-name swapdata_$INIT --out $I/swapdata_$INIT.pt 2>&1 | grep --line-buffered -v httpx
done

LENSES="$I/swapdata_jlens.pt $I/swapdata_identity.pt $I/plain_jlens64.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt logit"
echo "== swap success on held-out domains $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS --lens $LENSES \
    --data $D/heldout.json --out $I/swapdata_heldout_swap.parquet 2>&1 | grep --line-buffered -v httpx
echo "== probe-swap $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS --lens $LENSES \
    --out $I/swapdata_probe_swap.parquet 2>&1 | grep --line-buffered -v httpx
echo "== evaluate $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $LENSES --out $I/eval_held_out_swapdata.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
echo "SWAP_DATASET_DONE $(date +%T)"
