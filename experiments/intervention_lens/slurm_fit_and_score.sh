#!/bin/bash
# Runs 0-7 of the intervention-trained lens spec on the collected pairs, then scores every
# lens on the held-out items next to the J++ (k-means 8), R-Lens and Logit baselines.
#   0 output-KL (Tuned-Lens-style)    1 gauss              2 swap alpha 1 (main)
#   3 swap alpha 0.5 and 0.25         4 run 2, logit loss  5 run 2, init R-Lens
#   6 run 2, target t = s only        7 fit-weights over {run 2, swap 0.5, run 1, R-Lens}
# Needs slurm_collect.sh and slurm_plain_jlens.sh done.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J ilens-fit
#SBATCH -o logs/intervention_fit_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
CSV=$OUT/model_correctness_stripped.csv
RLENS=$OUT/pooled_lens.pt
fit() {  # fit <lens name> <family dir> [extra flags]
    local NAME=$1 FAMILY=$2; shift 2
    [ -f $I/$NAME.pt ] || $PY scripts/jpp_cli.py fit-intervention --layers $LAYERS \
        --pairs-dir $I/$FAMILY --checkpoint-name $NAME --out $I/$NAME.pt "$@"
}

echo "== runs 1-6 $(date +%T)"
fit ilens_gauss gauss_s0.05
fit ilens_swap_a1 swap_a1
fit ilens_swap_a0.5 swap_a0.5
fit ilens_swap_a0.25 swap_a0.25
fit ilens_swap_a1_logit swap_a1 --loss logit
fit ilens_swap_a1_init_rlens swap_a1 --init $RLENS
fit ilens_swap_a1_same_pos swap_a1 --target same_position

echo "== run 0 $(date +%T)"
[ -f $I/ilens_output_kl.pt ] || $PY scripts/jpp_cli.py fit-output-kl --hf-model-name $MODEL \
    --layers $LAYERS --pool-dir $I --checkpoint-name ilens_output_kl --out $I/ilens_output_kl.pt \
    2>&1 | grep --line-buffered -v httpx

echo "== compare maps $(date +%T)"
LENSES="$I/ilens_gauss.pt $I/ilens_swap_a1.pt $I/ilens_swap_a0.5.pt $I/ilens_swap_a0.25.pt \
    $I/ilens_swap_a1_logit.pt $I/ilens_swap_a1_init_rlens.pt $I/ilens_swap_a1_same_pos.pt \
    $I/ilens_output_kl.pt"
$PY experiments/intervention_lens/compare_lenses.py --layers $LAYERS --reference $I/plain_jlens64.pt \
    --lens $LENSES $I/plain_jlens64.pt $RLENS \
    --pairs-dirs $I/gauss_s0.05 $I/swap_a1 $I/swap_a0.5 $I/swap_a0.25 --out $I/compare_maps.csv

echo "== run 7 $(date +%T)"
$PY scripts/jpp_cli.py build-experts --layers $LAYERS \
    --lens $I/ilens_swap_a1.pt $I/ilens_swap_a0.5.pt $I/ilens_gauss.pt $RLENS --out $I/experts_ilens_mix.pt
$PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --experts $I/experts_ilens_mix.pt --checkpoint-name ilens_mix --out $I/ilens_mix.pt 2>&1 | grep --line-buffered -v httpx

echo "== evaluate $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $LENSES $I/ilens_mix.pt $I/plain_jlens64.pt $OUT/jpp_lens_v2.pt $RLENS logit \
    --out $I/eval_held_out.csv 2>&1 | grep --line-buffered -v httpx
echo "FIT_DONE $(date +%T)"
