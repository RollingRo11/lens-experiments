#!/bin/bash
# The plain-gradient J-lens on the first 64 WikiText prompts (the post's Table 5 size), the
# reference for the gauss sanity check: fit-shard with --lrp-mode none (standard gradients),
# then its pooled lens. The router only buckets the expert sums; the pooled lens ignores it.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J ilens-jlens64
#SBATCH -o logs/intervention_plain_jlens_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
echo "== fit-shard $(date +%T)"
$PY scripts/jpp_cli.py fit-shard --hf-model-name $MODEL --layers $LAYERS --router-path $OUT/router.pt \
    --num-prompts 64 --lrp-mode none --checkpoint-name plain_jlens64 \
    --artifacts-base-dir $OUT/intervention/shards 2>&1 | grep --line-buffered -v httpx | tail -5
SHARD_DIR=$(ls -d $OUT/intervention/shards/*/plain_jlens64/shard0of1 | tail -1)
echo "== merge-experts $(date +%T)"
$PY scripts/jpp_cli.py merge-experts --checkpoint-dirs $SHARD_DIR --checkpoint-name plain_jlens64 \
    --out $OUT/intervention/plain_jlens64_experts.pt --pooled-lens-out $OUT/intervention/plain_jlens64.pt
echo "PLAIN_JLENS_DONE $(date +%T)"
