#!/bin/bash
# Fix 5: the large swap dataset rebuilt with one canonical ID per relation (relation_ids.py)
# and the held-out relations drawn over those IDs (test 20%, dev 10% of the IDs with >= 20
# items), with each item's clean ranks recorded; the all-layer swap lens retrained on it
# (4,000 steps, batch 256, from identity, the swapall_v3 settings; dev relations as the W&B
# monitor); swap success per layer set on dev and test, with windows chosen on dev and
# reported on test as relation-level means; probe-swap over the band; and the J-lens
# first-order check at shrinking edit sizes (diagnose_check.py). Skips steps whose output exists.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 05:00:00
#SBATCH -J review-swap-v4
#SBATCH -o /projects/RohanKathuria/jpp_lens_review/logs/review_swap_v4_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens_review/experiments/review_fixes/common.sh
D=$R/swap_data_v4
X=data_ext
ALL=$(seq -s, 0 33)
BAND=$(seq -s, 8 26)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== build dataset $(date +%T)"
[ -f $D/test.json ] || $PY experiments/intervention_lens/build_swap_dataset_v2.py --ravel-dir $X/ravel \
    --lre-dir $X/relations/data --extra-relations $X/relations_extra/*.json \
    --templates-per-relation 20 --subjects-per-relation 1000 --pairs-per-template 300 \
    --heldout-fraction 0.2 --dev-fraction 0.1 --min-eval-pairs 20 --max-eval-items-per-relation 100 \
    --out-dir $D 2>&1 | quiet

echo "== fit $(date +%T)"
[ -f $R/swapall_v4.pt ] || $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $ALL \
    --mode supervised --swap-data $D/train.json --init identity --steps 4000 --batch-size 256 \
    --eval-every 250 --patience 1000 --heldout-examples 1024 --monitor-data $D/dev.json \
    --all-layers-probability 0.5 --window-sizes 1,3,5 \
    --wandb-project lens-training --checkpoint-name swapall_v4 --out $R/swapall_v4.pt 2>&1 | quiet

echo "== swap success per layer set, dev and test relations $(date +%T)"
[ -f $R/swapv4_windows.parquet ] || $PY experiments/intervention_lens/eval_layer_windows.py \
    --data dev=$D/dev.json test=$D/test.json --band 8-26 \
    --lens $R/swapall_v4.pt $I/plain_jlens64_all.pt logit:$ALL --out $R/swapv4_windows.parquet 2>&1 | quiet
$PY experiments/intervention_lens/select_windows.py --rows $R/swapv4_windows.parquet --out $R/swapv4_windows_selected.csv
$PY experiments/intervention_lens/select_windows.py --rows $R/swapv4_windows.parquet --clean-top1 \
    --out $R/swapv4_windows_selected_clean_top1.csv

echo "== probe-swap over the band $(date +%T)"
[ -f $R/swapall_v4_probe_swap_band.parquet ] || $PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL \
    --layers $BAND --lens $R/swapall_v4.pt $I/plain_jlens64_all.pt logit \
    --out $R/swapall_v4_probe_swap_band.parquet 2>&1 | grep --line-buffered -v -e httpx -e "ran trial"

echo "== first-order check at shrinking edit sizes $(date +%T)"
[ -f $R/diagnose_check_errors.csv ] || $PY experiments/intervention_lens/diagnose_check.py \
    --out $R/diagnose_check_errors.csv 2>&1 | quiet
echo "REVIEW_SWAP_V4_DONE $(date +%T)"
