#!/bin/bash
# Fix 1: refit only the expert weights of every J++ clustering lens and of the Part 3 mix with
# the fixed fitter (fp32 logits before Gemma's soft-cap, ties against the lens, pooled copies
# collapsed into one slot per layer), reusing the saved expert Jacobians. Then score every
# lens (old and refitted) with evaluate on the same held-out split, and the routed lenses.
# Skips a lens whose output file exists.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J review-refit
#SBATCH -o /projects/RohanKathuria/jpp_lens_review/logs/review_refit_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens_review/experiments/review_fixes/common.sh
# Own folder and residuals cache, so no other review job reads a file this one is writing.
F=$R/refit
mkdir -p $F
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

declare -A EXPERTS=(
    [kmeans8]=$OUT/experts.pt
    [kmeans_matched]=$OUT/experts_kmeans_matched.pt
    [adp]=$OUT/experts_adp.pt
    [adp_top7]=$OUT/experts_adp_top7.pt
    [adp_shuffled]=$OUT/experts_adp_shuffled.pt
    [hdbscan]=$OUT/experts_hdbscan.pt
    [ilens_mix_zero]=$I/experts_ilens_mix_zero.pt
)
for NAME in kmeans8 kmeans_matched adp adp_top7 adp_shuffled hdbscan ilens_mix_zero; do
    [ -f $F/jpp_$NAME.pt ] && { echo "skip $NAME (exists)"; continue; }
    echo "== fit-weights $NAME $(date +%T)"
    $PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
        --experts ${EXPERTS[$NAME]} --checkpoint-name gemma4_e2b_jpp_${NAME}_refit \
        --cache-path $F/readout_residuals.pt --out $F/jpp_$NAME.pt 2>&1 | quiet
done

NEW="$F/jpp_kmeans8.pt $F/jpp_kmeans_matched.pt $F/jpp_adp.pt $F/jpp_adp_top7.pt $F/jpp_adp_shuffled.pt \
    $F/jpp_hdbscan.pt $F/jpp_ilens_mix_zero.pt"
OLD="$OUT/jpp_lens_v2.pt $OUT/jpp_lens_kmeans_matched.pt $OUT/jpp_lens_adp.pt $OUT/jpp_lens_adp_top7.pt \
    $OUT/jpp_lens_adp_shuffled.pt $OUT/jpp_lens_hdbscan.pt $I/ilens_mix_zero.pt"
BASE="$OUT/pooled_lens.pt logit $I/plain_jlens64.pt"
PART3="$I/ilens_gauss.pt $I/ilens_output_kl.pt $I/ilens_swap_a1_zero.pt $I/ilens_swap_a1.pt $I/ilens_swap_a0.5.pt \
    $I/ilens_swap_a0.25.pt $I/ilens_swap_a1_logit.pt $I/ilens_swap_a1_init_rlens.pt"
PART4="$I/swapdata_identity.pt $I/swapdata_jlens.pt $I/swapdata2_identity.pt $I/swapdata2_jlens.pt \
    $I/swapall_v3.pt $I/swapall_v3_16000.pt"
if [ ! -f $F/eval_held_out_rescore.csv ]; then
    echo "== evaluate $(date +%T)"
    $PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
        --lens $NEW $OLD $BASE $PART3 $PART4 --ks 1,5,10,20 --out $F/eval_held_out_rescore.csv 2>&1 | quiet
fi
if [ ! -f $F/eval_held_out_routed.csv ]; then
    echo "== routed lenses $(date +%T)"
    $PY experiments/routed_lens/eval_routed.py --out $F/eval_held_out_routed.csv 2>&1 | quiet
fi
echo "REVIEW_REFIT_DONE $(date +%T)"
