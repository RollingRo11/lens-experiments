#!/bin/bash
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -J adp-prep
#SBATCH -o logs/adp_prep_%j.log
set -euo pipefail
cd /projects/RohanKathuria/jpp_lens
mkdir -p results/adp_router
.venv/bin/python -u experiments/adp_router/prep_points.py
