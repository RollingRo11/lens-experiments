#!/bin/bash
# ADP on each layer's 20k raw (1536-dim) activations, one array task per layer (CPU).
#SBATCH -p short
#SBATCH --array=0-6
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J adp-zsweep
#SBATCH -o logs/adp_zsweep_%a.log
set -euo pipefail
cd /projects/RohanKathuria/jpp_lens
LAYERS=(4 8 12 16 20 24 28)
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
export UV_BUILD_CONSTRAINT=experiments/dimensionality/build-constraints.txt
~/.local/bin/uv run --no-project --python 3.12 --with "numpy<2" --with dadapy --with scikit-learn \
    python -u experiments/adp_router/adp_z_sweep.py --layer ${LAYERS[$SLURM_ARRAY_TASK_ID]}
