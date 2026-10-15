#!/bin/bash
# PCA spectrum and TwoNN intrinsic dimension of the saved router activations.
# A100 only (H200 also works): the venv's torch has no kernels for the V100s (compute capability 7.0).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J pca-twonn
#SBATCH -o logs/pca_and_twonn_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
$PY experiments/dimensionality/pca_and_twonn.py
echo PCA_TWONN_DONE
