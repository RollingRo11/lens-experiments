#!/bin/bash
# Runs 2-7 again with removal edits (zero-ablation, delta = -h) pooled into the swap pairs, so
# the maps also see the average part of the residual that swap edits never contain; plus the
# removal edits alone. Scores them next to the first round's lenses and the baselines.
# Needs the zero family collected (slurm_collect.sh with FAMILIES=zero).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J ilens-removal
#SBATCH -o logs/intervention_removal_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
CSV=$OUT/model_correctness_stripped.csv
RLENS=$OUT/pooled_lens.pt
fit() {  # fit <lens name> <family dirs...> -- [extra flags]
    local NAME=$1; shift
    local DIRS=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do DIRS+=("$I/$1"); shift; done
    [ $# -gt 0 ] && shift
    [ -f $I/$NAME.pt ] || $PY scripts/jpp_cli.py fit-intervention --layers $LAYERS \
        --pairs-dir "${DIRS[@]}" --checkpoint-name $NAME --out $I/$NAME.pt "$@"
}

echo "== fits $(date +%T)"
fit ilens_zero zero
fit ilens_swap_a1_zero swap_a1 zero
fit ilens_swap_a0.5_zero swap_a0.5 zero
fit ilens_swap_a0.25_zero swap_a0.25 zero
fit ilens_swap_a1_zero_logit swap_a1 zero -- --loss logit
fit ilens_swap_a1_zero_init_rlens swap_a1 zero -- --init $RLENS
fit ilens_swap_a1_zero_same_pos swap_a1 zero -- --target same_position

echo "== compare maps $(date +%T)"
LENSES="$I/ilens_zero.pt $I/ilens_swap_a1_zero.pt $I/ilens_swap_a0.5_zero.pt $I/ilens_swap_a0.25_zero.pt \
    $I/ilens_swap_a1_zero_logit.pt $I/ilens_swap_a1_zero_init_rlens.pt $I/ilens_swap_a1_zero_same_pos.pt"
$PY experiments/intervention_lens/compare_lenses.py --layers $LAYERS --reference $I/plain_jlens64.pt \
    --lens $LENSES $I/ilens_swap_a1.pt $I/plain_jlens64.pt \
    --pairs-dirs $I/swap_a1 $I/zero --out $I/compare_maps_removal.csv

echo "== mix $(date +%T)"
$PY scripts/jpp_cli.py build-experts --layers $LAYERS \
    --lens $I/ilens_swap_a1_zero.pt $I/ilens_swap_a0.5_zero.pt $I/ilens_gauss.pt $RLENS \
    --out $I/experts_ilens_mix_zero.pt
$PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --experts $I/experts_ilens_mix_zero.pt --checkpoint-name ilens_mix_zero --out $I/ilens_mix_zero.pt \
    2>&1 | grep --line-buffered -v httpx

echo "== evaluate $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $LENSES $I/ilens_mix_zero.pt $I/ilens_swap_a1.pt $I/ilens_gauss.pt $I/plain_jlens64.pt \
    $OUT/jpp_lens_v2.pt $RLENS logit \
    --out $I/eval_held_out_removal.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
echo "REMOVAL_DONE $(date +%T)"
