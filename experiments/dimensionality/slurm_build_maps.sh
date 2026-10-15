#!/bin/bash
# datamapplot maps for every layer x clustering, plus clusterings.json for the 3D page (CPU only).
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -J build-maps
#SBATCH -o logs/build_maps_%j.log
set -euo pipefail
cd /projects/RohanKathuria/jpp_lens
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
~/.local/bin/uv run --no-project --python 3.12 --with datamapplot \
    python -u experiments/dimensionality/build_maps.py
