#!/bin/bash
# WorkspaceBench's single-token families (association, multihop, multilingual, poetry, typo)
# scored with our recall@k instead of its LLM judge, on every lens we have: items in our J++
# fitting split dropped, Gemma-correctness filter graded as for model_correctness_stripped.csv.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J wsbench-eval
#SBATCH -o logs/wsbench_eval_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
W=data_ext/workspace-bench
B=$OUT/workspace_bench
E=$B/eval_data
mkdir -p $B
[ -d $W ] || git clone --depth 1 -q https://github.com/camilablank/workspace-bench $W
[ -f $B/fit_split_prompts.json ] || $PY experiments/workspace_bench/fit_split_prompts.py \
    --correctness-csv $OUT/model_correctness_stripped.csv --out $B/fit_split_prompts.json 2>&1 | grep --line-buffered -v httpx
$PY experiments/workspace_bench/convert_banks.py --wsbench $W --exclude-prompts $B/fit_split_prompts.json --out-dir $E
[ -f $E/model_correctness.csv ] || $PY scripts/grade_correctness.py --hf-model-name $MODEL --eval-data-dir $E \
    --correctness-csv $E/correctness_seed.csv --strip-trailing-space --out $E/model_correctness.csv 2>&1 | grep --line-buffered -v httpx
LENSES="logit $OUT/pooled_lens.pt $OUT/jpp_lens_v2.pt $OUT/jpp_lens_kmeans_matched.pt $OUT/jpp_lens_adp.pt \
    $OUT/jpp_lens_adp_top7.pt $OUT/jpp_lens_hdbscan.pt $I/plain_jlens64.pt $I/ilens_gauss.pt \
    $I/ilens_swap_a1_zero.pt $I/ilens_mix_zero.pt $I/ilens_output_kl.pt $I/swapdata_jlens.pt $I/swapdata_identity.pt"
for EXTRA in $I/swapdata2_jlens.pt $I/swapdata2_identity.pt $I/swapall_identity.pt; do
    [ -f $EXTRA ] && LENSES="$LENSES $EXTRA"
done
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --eval-data-dir $E \
    --correctness-csv $E/model_correctness.csv --items all --lens $LENSES \
    --out $B/eval_all_lenses.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
echo "WSBENCH_EVAL_DONE $(date +%T)"
