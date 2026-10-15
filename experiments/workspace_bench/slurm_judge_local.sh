#!/bin/bash
# Judge our lenses' WorkspaceBench readouts with a local judge: Qwen3.6-27B served by vLLM on
# this GPU (OpenAI-compatible), wsbench patched to use it instead of OpenRouter
# (local_judge.patch). Judge numbers are consistent across our lenses but not comparable to the
# published Gemini-judged ones. Needs slurm_produce.sh's readouts.
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -J wsbench-judge
#SBATCH -o logs/wsbench_judge_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
UV=~/.local/bin/uv
W=$PWD/data_ext/workspace-bench
B=$OUT/workspace_bench
R=$B/readouts
J=$B/judged_qwen3.6-27b
JUDGE=Qwen/Qwen3.6-27B
PORT=8765
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $W && (git apply --check $OLDPWD/experiments/workspace_bench/local_judge.patch 2>/dev/null \
    && git apply $OLDPWD/experiments/workspace_bench/local_judge.patch || echo "patch already applied") && cd $OLDPWD

echo "== vLLM $(date +%T)"
$UV run --no-project --python 3.12 --with vllm vllm serve $JUDGE --served-model-name $JUDGE --port $PORT \
    --max-model-len 16384 --gpu-memory-utilization 0.88 > logs/wsbench_vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT
for i in $(seq 1 120); do
    curl -sf localhost:$PORT/v1/models >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited"; tail -40 logs/wsbench_vllm_${SLURM_JOB_ID}.log; exit 1; }
    sleep 30
done
curl -sf localhost:$PORT/v1/models >/dev/null || { echo "vLLM not up after 60 min"; exit 1; }
echo "== vLLM up $(date +%T)"

export WSBENCH_BASE_URL=http://localhost:$PORT/v1
WS="$UV run --project $W --extra dev wsbench"
FAMILIES=$(ls $R/logit | sed 's/\.jsonl$//' | paste -sd, -)
echo "families: $FAMILIES"
echo "== smoke $(date +%T)"
$WS judge family=multihop readouts=$R/logit/multihop.jsonl out=$J/_smoke/multihop judge_model=$JUDGE limit=3 \
    allow_missing=True

for LENS in logit pooled_lens jpp_lens_v2 plain_jlens64 swapdata_jlens swapdata_identity ilens_gauss \
        ilens_swap_a1_zero ilens_mix_zero ilens_output_kl jpp_lens_kmeans_matched jpp_lens_adp \
        jpp_lens_adp_top7 jpp_lens_hdbscan swapdata2_jlens swapdata2_identity swapall_identity; do
    [ -d $R/$LENS ] || continue
    echo "== judge $LENS $(date +%T)"
    $WS run families=$FAMILIES readouts_root=$R/$LENS out=$J/$LENS judge_model=$JUDGE \
        concurrency=96 rpm=100000 allow_missing=True || echo "run for $LENS exited $?"
    $WS report dir=$J/$LENS || true
done
echo "WSBENCH_JUDGE_DONE $(date +%T)"
