#!/bin/bash
# fit-weights + evaluate only, for a variant whose experts file exists (H200: 191-expert weight
# fits need more than a 40 GB A100).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J jpp-weights-variant
#SBATCH -o logs/jpp_weights_%x_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
CSV=$OUT/model_correctness_stripped.csv
$PY scripts/jpp_cli.py fit-weights --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --experts $OUT/experts_$NAME.pt --checkpoint-name gemma4_e2b_jpp_$NAME --out $OUT/jpp_lens_$NAME.pt
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $OUT/jpp_lens_$NAME.pt $OUT/jpp_lens_adp.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt logit \
    --out $OUT/eval_held_out_$NAME.csv
echo VARIANT_DONE
