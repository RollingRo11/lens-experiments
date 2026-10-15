#!/bin/bash
# WorkspaceBench readouts for every lens we have on Gemma 4 E2B (no judging: that needs an
# OpenRouter key). Clones github.com/camilablank/workspace-bench into data_ext/ once.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J wsbench-produce
#SBATCH -o logs/wsbench_produce_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
W=data_ext/workspace-bench
[ -d $W ] || git clone --depth 1 -q https://github.com/camilablank/workspace-bench $W
export PYTHONPATH=$W/src
LENSES="logit $OUT/pooled_lens.pt $OUT/jpp_lens_v2.pt $OUT/jpp_lens_kmeans_matched.pt $OUT/jpp_lens_adp.pt \
    $OUT/jpp_lens_adp_top7.pt $OUT/jpp_lens_hdbscan.pt $I/plain_jlens64.pt $I/ilens_gauss.pt \
    $I/ilens_swap_a1_zero.pt $I/ilens_mix_zero.pt $I/ilens_output_kl.pt $I/swapdata_jlens.pt $I/swapdata_identity.pt"
for EXTRA in $I/swapdata2_jlens.pt $I/swapdata2_identity.pt $I/swapall_identity.pt; do
    [ -f $EXTRA ] && LENSES="$LENSES $EXTRA"
done
echo "lenses: $LENSES"
$PY experiments/workspace_bench/produce_readouts.py --lens $LENSES --out-dir $OUT/workspace_bench/readouts \
    2>&1 | grep --line-buffered -v httpx
du -sh $OUT/workspace_bench/readouts
echo "WSBENCH_PRODUCE_DONE $(date +%T)"
