#!/bin/bash
# Fix 4: random-partition control. One Jacobian fit pass (the same 1,024 WikiText prompts and
# LRP recipe as every J++ fit here) accumulating nine partitions of the fit positions:
# random buckets of equal size (K = 8 and 32, seeds 0-2), k-means K = 8 and 32 on ADP's 20k
# points, and one pooled bucket. Then merge each, fit its weights with the fixed fitter and
# score them all on the held-out split next to the refitted J++ lenses. Skips steps whose
# output exists.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:30:00
#SBATCH -J review-partitions
#SBATCH -o /projects/RohanKathuria/jpp_lens_review/logs/review_partitions_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens_review/experiments/review_fixes/common.sh
P=$R/partitions
mkdir -p $P
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NAMES="random8_s0 random8_s1 random8_s2 random32_s0 random32_s1 random32_s2 kmeans20k_K8 kmeans20k_K32 pooled_K1"

echo "== routers $(date +%T)"
[ -f $P/router_pooled_K1.pt ] || $PY experiments/review_fixes/build_partition_routers.py --out-dir $P 2>&1 | quiet

SHARD=$P/shards/fit/gemma4_e2b_partitions/shard0of1
if [ ! -f $SHARD/experts_pooled_K1_checkpoint.pt ] || [ ! -f $P/fit_done ]; then
    echo "== fit $(date +%T)"
    ROUTERS=""
    for NAME in $NAMES; do ROUTERS="$ROUTERS $NAME=$P/router_$NAME.pt"; done
    $PY scripts/jpp_cli.py fit-shard --hf-model-name $MODEL --layers $LAYERS --router-path $ROUTERS \
        --num-prompts 1024 --checkpoint-every-n-prompts 128 --checkpoint-name gemma4_e2b_partitions \
        --artifacts-base-dir $P/shards 2>&1 | quiet
    # fit-shard writes under <base>/<date>/<name>; give it a fixed path.
    mkdir -p $P/shards/fit
    [ -e $P/shards/fit/gemma4_e2b_partitions ] || ln -s "$(ls -d $P/shards/20*/gemma4_e2b_partitions | tail -1)" $P/shards/fit/gemma4_e2b_partitions
    touch $P/fit_done
fi

for NAME in $NAMES; do
    [ -f $P/experts_$NAME.pt ] || $PY scripts/jpp_cli.py merge-experts --checkpoint-dirs $SHARD --partition $NAME \
        --checkpoint-name gemma4_e2b_$NAME --out $P/experts_$NAME.pt 2>&1 | quiet
done
for NAME in $NAMES; do
    [ -f $P/jpp_$NAME.pt ] && continue
    echo "== fit-weights $NAME $(date +%T)"
    $PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
        --experts $P/experts_$NAME.pt --checkpoint-name gemma4_e2b_jpp_$NAME \
        --cache-path $R/readout_residuals_all.pt --out $P/jpp_$NAME.pt 2>&1 | quiet
done

LENSES=""
for NAME in $NAMES; do LENSES="$LENSES $P/jpp_$NAME.pt"; done
for EXTRA in $R/jpp_kmeans8.pt $R/jpp_adp.pt $R/jpp_adp_shuffled.pt; do [ -f $EXTRA ] && LENSES="$LENSES $EXTRA"; done
echo "== evaluate $(date +%T)"
[ -f $P/eval_held_out_partitions.csv ] || $PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS \
    --correctness-csv $CSV --lens $LENSES $OUT/pooled_lens.pt logit --ks 1,5,10,20 \
    --out $P/eval_held_out_partitions.csv 2>&1 | quiet
echo "REVIEW_PARTITIONS_DONE $(date +%T)"
