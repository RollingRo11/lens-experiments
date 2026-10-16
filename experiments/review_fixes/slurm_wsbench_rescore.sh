#!/bin/bash
# Fix 3: WorkspaceBench's five single-token families rescored after dropping every prompt
# that is also one of our own questions (all our question files, not only the J++ fitting
# split; whitespace- and case-normalised match), graded for Gemma as for our questions
# (trailing space stripped), on every lens. Skips steps whose output exists.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J review-wsbench
#SBATCH -o /projects/RohanKathuria/jpp_lens_review/logs/review_wsbench_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens_review/experiments/review_fixes/common.sh
W=$R/workspace_bench
E=$W/eval_data
F=$R/refit
P=$R/partitions
mkdir -p $W
$PY experiments/workspace_bench/convert_banks.py --wsbench data_ext/workspace-bench --out-dir $E
[ -f $E/model_correctness.csv ] || $PY scripts/grade_correctness.py --hf-model-name $MODEL --eval-data-dir $E \
    --correctness-csv $E/correctness_seed.csv --strip-trailing-space --out $E/model_correctness.csv 2>&1 | quiet
LENSES="logit $OUT/pooled_lens.pt $I/plain_jlens64.pt $I/ilens_gauss.pt $I/ilens_output_kl.pt $I/ilens_swap_a1_zero.pt \
    $I/swapdata_identity.pt $I/swapdata_jlens.pt $I/swapdata2_identity.pt $I/swapdata2_jlens.pt $I/swapall_identity.pt \
    $I/swapall_v3.pt $I/swapall_v3_16000.pt"
for EXTRA in $F/jpp_kmeans8.pt $F/jpp_kmeans_matched.pt $F/jpp_adp.pt $F/jpp_adp_top7.pt $F/jpp_adp_shuffled.pt \
    $F/jpp_hdbscan.pt $F/jpp_ilens_mix_zero.pt $R/swapall_v4.pt $P/jpp_random8_s0.pt $P/jpp_random32_s0.pt \
    $P/jpp_kmeans20k_K8.pt $P/jpp_kmeans20k_K32.pt $P/jpp_pooled_K1.pt; do
    [ -f $EXTRA ] && LENSES="$LENSES $EXTRA"
done
[ -f $W/eval_all_lenses.csv ] || $PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS \
    --eval-data-dir $E --correctness-csv $E/model_correctness.csv --items all --lens $LENSES --ks 1,10 \
    --out $W/eval_all_lenses.csv 2>&1 | quiet
echo "REVIEW_WSBENCH_DONE $(date +%T)"
