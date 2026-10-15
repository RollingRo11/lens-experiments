#!/bin/bash
# J++ lens with the ADP router: build the router, fit expert Jacobians (one job, 1024 prompts),
# merge, fit expert weights, then score it next to the k-means J++ lens on the same held-out items.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -J jpp-adp
#SBATCH -o logs/jpp_adp_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
CSV=$OUT/model_correctness_stripped.csv   # same grading as the k-means v2 run
$PY experiments/adp_router/build_router.py
$PY scripts/jpp_cli.py fit-shard --hf-model-name $MODEL --layers $LAYERS --router-path $OUT/router_adp.pt \
    --num-prompts 1024 --checkpoint-name gemma4_e2b_adp --artifacts-base-dir $OUT/shards
$PY scripts/jpp_cli.py merge-experts --checkpoint-dirs $OUT/shards/*/gemma4_e2b_adp/shard0of1 \
    --num-clusters $($PY -c "import torch;print(torch.load('$OUT/router_adp.pt',weights_only=True)['num_clusters'])") \
    --checkpoint-name gemma4_e2b_adp --out $OUT/experts_adp.pt --pooled-lens-out $OUT/pooled_lens_adp.pt
$PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --experts $OUT/experts_adp.pt --checkpoint-name gemma4_e2b_jpp_adp --out $OUT/jpp_lens_adp.pt
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $OUT/jpp_lens_adp.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt logit --out $OUT/eval_held_out_adp.csv
echo ADP_DONE
