#!/bin/bash
# One experiment / one fold on PARCC Betty.
# https://parcc.upenn.edu/training/slurm/
#
#   sbatch --export=EXP=P3_stage1_oldpenn_b,FOLD=0 experiments/slurm/train.sbatch
#   sbatch --export=EXP=P3_staged_s,FOLD=0 experiments/slurm/train.sbatch
#
# ViT-S is fine on b200-mig45 (45 GB). Use b200-mig90 for ViT-B/L.
# Full dgx-b200 (180 GB GPU) is more than this recipe needs.
#
#SBATCH --job-name=mst
#SBATCH --partition=b200-mig90
#SBATCH --gpus=1
#SBATCH --ntasks=1
# Betty CLI filter (Aug 2026): b200-mig90 is 14 CPUs/GPU and 8 GB/CPU (112 GB/GPU).
# https://docs.parcc.upenn.edu/docs/notes/filter.html
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/slurm/%x_%j.out
#SBATCH --error=experiments/logs/slurm/%x_%j.err

set -euo pipefail

# Code is on project VAST. Conda env `mst` stays in $HOME.
# Data (h5 + CSVs) and run folders: also project VAST, not Ceph and not $HOME.
# https://parcc.upenn.edu/training/storage/
MST_ROOT="${MST_ROOT:-/vast/projects/bbruno/breast-imaging/arthur/MST}"
CONDA_ENV="${CONDA_ENV:-mst}"
EXP="${EXP:?set EXP, e.g. sbatch --export=EXP=P3_staged_s,FOLD=0 ...}"
FOLD="${FOLD:?set FOLD, e.g. 0}"

cd "$MST_ROOT"
mkdir -p experiments/logs/slurm

# Activate the conda environment.
if [[ -f "$MST_ROOT/.venv/bin/activate" ]]; then
  source "$MST_ROOT/.venv/bin/activate"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# If the .h5 still lives on Ceph, copy it to the node's NVMe for this job and
# point machines.cluster.data_root at that folder (or export MST_DATA_ROOT if
# you switch the plan to ${MST_DATA_ROOT}).
# mkdir -p "/tmp/$USER/MST_birads4"
# cp -n "$CEPH_DATA"/*.h5 "$CEPH_DATA"/*.csv "/tmp/$USER/MST_birads4/"

echo "host=$(hostname)  cuda=$(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "exp=$EXP fold=$FOLD"

python scripts/experiments/run_plan.py run \
  --machine cluster \
  --exp "$EXP" \
  --folds "$FOLD"
