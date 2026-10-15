#!/bin/bash
# J++ lens for one router variant: build router, fit expert Jacobians (one job, 1024 prompts),
# merge, fit weights, score next to the k-means and 191-expert ADP J++ lenses.
# Env: NAME (e.g. adp_top7), ROUTER_ARGS (args for the builder, minus --out),
# BUILDER (default build_router_variant.py), FIT_ARGS (extra fit-shard args).
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -J jpp-variant
#SBATCH -o logs/jpp_%x_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
CSV=$OUT/model_correctness_stripped.csv   # same grading as the k-means v2 run
ROUTER=$OUT/router_$NAME.pt
$PY experiments/adp_router/${BUILDER:-build_router_variant.py} ${ROUTER_ARGS:-} --out $ROUTER
K=$($PY -c "import torch;print(torch.load('$ROUTER',weights_only=True)['num_clusters'])")
$PY scripts/jpp_cli.py fit-shard --hf-model-name $MODEL --layers $LAYERS --router-path $ROUTER \
    --num-prompts 1024 --checkpoint-name gemma4_e2b_$NAME --artifacts-base-dir $OUT/shards ${FIT_ARGS:-}
$PY scripts/jpp_cli.py merge-experts --checkpoint-dirs $OUT/shards/*/gemma4_e2b_$NAME/shard0of1 \
    --num-clusters $K --checkpoint-name gemma4_e2b_$NAME --out $OUT/experts_$NAME.pt
$PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --experts $OUT/experts_$NAME.pt --checkpoint-name gemma4_e2b_jpp_$NAME --out $OUT/jpp_lens_$NAME.pt
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $OUT/jpp_lens_$NAME.pt $OUT/jpp_lens_adp.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt logit \
    --out $OUT/eval_held_out_$NAME.csv
echo VARIANT_DONE
