#!/bin/bash
# HDBSCAN sweep on all 7 layers' 20k raw points (CPU).
#SBATCH -p short
#SBATCH -c 16
#SBATCH --array=0-5
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J hdbscan-router
#SBATCH -o logs/hdbscan_router_%a.log
set -euo pipefail
cd /projects/RohanKathuria/jpp_lens
mkdir -p results/adp_router/hdbscan
LAYERS=(8 12 16 20 24 28)  # layer 4 finished in the first (all-layer) run
export UV_BUILD_CONSTRAINT=experiments/dimensionality/build-constraints.txt
~/.local/bin/uv run --no-project --python 3.12 --with "numpy<2" --with scikit-learn \
    python -u experiments/adp_router/hdbscan_layers.py --layer ${LAYERS[$SLURM_ARRAY_TASK_ID]}
