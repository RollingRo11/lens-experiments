#!/bin/bash
# PCA, t-SNE and tuned UMAP (2D + 3D) for one layer per array task (CPU only).
# Runs in its own uv env (numpy/scipy/scikit-learn/umap-learn), not the project venv.
#SBATCH -p short
#SBATCH --array=0-6
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J embed-layer
#SBATCH -o logs/embed_layer_%a.log
set -euo pipefail
cd /projects/RohanKathuria/jpp_lens
LAYERS=(4 8 12 16 20 24 28)
# numba's compile cache otherwise sits in the shared uv cache, where concurrent tasks
# race on it over NFS ("Stale file handle"); give each task its own node-local copy.
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
~/.local/bin/uv run --no-project --python 3.12 --with umap-learn --with scikit-learn \
    python -u experiments/dimensionality/embed_layer.py --layer ${LAYERS[$SLURM_ARRAY_TASK_ID]}
