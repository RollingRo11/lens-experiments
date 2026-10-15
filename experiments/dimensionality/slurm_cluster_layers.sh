#!/bin/bash
# ADP and HDBSCAN on one layer's 20k displayed points per array task (CPU only).
# Own uv env with numpy<2: DADApy's C code only builds against numpy 1.x.
#SBATCH -p short
#SBATCH --array=0-6
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J cluster-layer
#SBATCH -o logs/cluster_layer_%a.log
set -euo pipefail
cd /projects/RohanKathuria/jpp_lens
LAYERS=(4 8 12 16 20 24 28)
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
export UV_BUILD_CONSTRAINT=experiments/dimensionality/build-constraints.txt
~/.local/bin/uv run --no-project --python 3.12 --with "numpy<2" --with dadapy --with scikit-learn \
    python -u experiments/dimensionality/cluster_layer.py --layer ${LAYERS[$SLURM_ARRAY_TASK_ID]}
