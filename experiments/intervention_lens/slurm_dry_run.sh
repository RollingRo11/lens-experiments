#!/bin/bash
# Dry run of the intervention-trained lens on 16 prompts: collect all four edit families,
# check the pairs against the J-lens's first-order prediction (does the replace hook work on
# Gemma?), fit two lenses and the Tuned-Lens-style lens briefly, and score them with the
# unchanged evaluate stage. Prints the timings that size the full job.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J ilens-dry
#SBATCH -o logs/intervention_dry_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
DRY=$OUT/intervention_dry
CSV=$OUT/model_correctness_stripped.csv
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== collect-pairs $(date +%T)"
$PY scripts/jpp_cli.py collect-pairs --hf-model-name $MODEL --layers $LAYERS --num-prompts 16 \
    --families gauss:0.05,swap:1,swap:0.5,swap:0.25 --pairs-per-prompt 32 --rows-per-pass 128 \
    --out-dir $DRY 2>&1 | grep --line-buffered -v httpx
du -sh $DRY/*

echo "== check-pairs $(date +%T)"
$PY experiments/intervention_lens/check_pairs.py --pairs-dir $DRY \
    --families gauss_s0.05,swap_a1,swap_a0.25 2>&1 | grep --line-buffered -v httpx

echo "== fit-intervention $(date +%T)"
for FAMILY in gauss_s0.05 swap_a1; do
    $PY scripts/jpp_cli.py fit-intervention --layers $LAYERS --pairs-dir $DRY/$FAMILY \
        --steps 500 --holdout-fraction 0.25 --checkpoint-name dry_$FAMILY --out $DRY/lens_$FAMILY.pt
done
$PY scripts/jpp_cli.py fit-intervention --layers $LAYERS --pairs-dir $DRY/swap_a1 --loss logit \
    --steps 500 --holdout-fraction 0.25 --checkpoint-name dry_swap_a1_logit --out $DRY/lens_swap_a1_logit.pt

echo "== fit-output-kl $(date +%T)"
$PY scripts/jpp_cli.py fit-output-kl --hf-model-name $MODEL --layers $LAYERS --pool-dir $DRY \
    --steps 200 --holdout-fraction 0.25 --checkpoint-name dry_output_kl --out $DRY/lens_output_kl.pt \
    2>&1 | grep --line-buffered -v httpx

echo "== evaluate $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $DRY/lens_swap_a1.pt $DRY/lens_output_kl.pt $OUT/pooled_lens.pt logit \
    --out $DRY/eval_held_out.csv 2>&1 | grep --line-buffered -v httpx
echo "DRY_DONE $(date +%T)"
