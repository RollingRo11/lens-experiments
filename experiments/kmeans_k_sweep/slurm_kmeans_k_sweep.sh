#!/bin/bash
# Re-record the router's activations, save them, and sweep k-means K on the 64-dim projection.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J kmeans-k-sweep
#SBATCH -o logs/kmeans_k_sweep_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
$PY experiments/kmeans_k_sweep/extract_and_sweep.py
echo SWEEP_DONE
