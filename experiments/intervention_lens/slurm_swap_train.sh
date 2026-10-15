#!/bin/bash
# The swap-loss lenses on the 1,024 WikiText prompts: both modes from the plain J-lens, and
# the counterfactual mode from the identity; then probe-swap and recall@10 against the
# baselines. STEPS sets the length of each fit.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J ilens-swaptrain
#SBATCH -o logs/intervention_swap_train_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
CSV=$OUT/model_correctness_stripped.csv
STEPS=${STEPS:-2000}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
swap() {  # swap <lens name> <mode> <init>
    [ -f $I/$1.pt ] || $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $LAYERS \
        --mode $2 --init $3 --steps $STEPS --checkpoint-name $1 --out $I/$1.pt 2>&1 | grep --line-buffered -v httpx
}
echo "== fits $(date +%T)"
swap swaplens_cf_jlens counterfactual $I/plain_jlens64.pt
swap swaplens_nt_jlens next_token $I/plain_jlens64.pt
swap swaplens_cf_identity counterfactual identity

LENSES="$I/swaplens_cf_jlens.pt $I/swaplens_nt_jlens.pt $I/swaplens_cf_identity.pt \
    $I/plain_jlens64.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt logit"
echo "== probe-swap $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS --lens $LENSES \
    --out $I/probe_swap_swaplens.parquet 2>&1 | grep --line-buffered -v httpx
echo "== evaluate $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $LENSES --out $I/eval_held_out_swaplens.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
echo "SWAP_TRAIN_DONE $(date +%T)"
