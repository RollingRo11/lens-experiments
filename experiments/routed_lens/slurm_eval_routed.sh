#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J jpp-routed
#SBATCH -o logs/jpp_routed_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
$PY experiments/routed_lens/eval_routed.py ${EVAL_ARGS:-} 2>&1 | grep --line-buffered -v httpx
