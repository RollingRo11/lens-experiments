#!/bin/bash
# Inputs for the per-layer embeddings (rows, 64-dim points, router clusters, hover labels).
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J embed-prep
#SBATCH -o logs/embed_prep_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
$PY experiments/dimensionality/prep_embed_inputs.py
