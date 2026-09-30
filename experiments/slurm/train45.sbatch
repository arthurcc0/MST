#!/bin/bash
# One experiment / one fold on PARCC Betty.
#
# Example:
#   sbatch --export=EXP=P3_stage1_oldpenn_b,FOLD=0 experiments/slurm/train.sbatch
#   sbatch --export=EXP=P3_staged_s,FOLD=0 experiments/slurm/train.sbatch
#
# ViT-S fits on b200-mig45 (45 GB).
# ViT-B/L should use b200-mig90.
#
#SBATCH --job-name=mst
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/slurm/%x_%j.out
#SBATCH --error=experiments/logs/slurm/%x_%j.err

set -euo pipefail

MST_ROOT="${MST_ROOT:-/vast/projects/bbruno/breast-imaging/arthur/MST}"
CONDA_ENV="${CONDA_ENV:-mst}"
EXP="${EXP:?set EXP, e.g. sbatch --export=EXP=P3_staged_s,FOLD=0 ...}"
FOLD="${FOLD:?set FOLD, e.g. sbatch --export=EXP=P3_staged_s,FOLD=0 ...}"

cd "$MST_ROOT"
mkdir -p experiments/logs/slurm

# Load the CUDA toolkit used to build/run the cluster environment.
module load cuda/12.8.1

# Activate the project environment.
if [[ -f "$MST_ROOT/.venv/bin/activate" ]]; then
    source "$MST_ROOT/.venv/bin/activate"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

echo "host=$(hostname)"
echo "cuda=$(python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "exp=$EXP fold=$FOLD"

python scripts/experiments/run_plan.py run \
    --machine cluster \
    --exp "$EXP" \
    --folds "$FOLD"