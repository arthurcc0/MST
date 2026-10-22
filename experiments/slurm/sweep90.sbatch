#!/bin/bash
# One W&B sweep *trial* on PARCC Betty (b200-mig90). Does not use EXP/FOLD.
# Create the sweep first (UI or `wandb sweep`), then:
#
#   sbatch --export=SWEEP=<entity>/Classifier_PENN_MST/<sweep_id> experiments/slurm/sweep90.sbatch
#
# COUNT=1 (default): this job trains one fold-0 trial and exits.
# Submit several jobs for parallel agents. Need WANDB_API_KEY in the env.
#
#SBATCH --job-name=mst-sweep
#SBATCH --partition=b200-mig90
#SBATCH --gpus=1
#SBATCH --ntasks=1
# Betty CLI filter (Aug 2026): b200-mig90 is 14 CPUs/GPU and 8 GB/CPU (112 GB/GPU).
# https://docs.parcc.upenn.edu/docs/notes/filter.html
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=6:00:00
#SBATCH --output=experiments/logs/slurm/%x_%j.out
#SBATCH --error=experiments/logs/slurm/%x_%j.err

set -euo pipefail

MST_ROOT="${MST_ROOT:-/vast/projects/bbruno/breast-imaging/arthur/MST}"
CONDA_ENV="${CONDA_ENV:-mst}"
SWEEP="${SWEEP:?set SWEEP, e.g. sbatch --export=SWEEP=entity/Classifier_PENN_MST/xxxxx ...}"
COUNT="${COUNT:-1}"

cd "$MST_ROOT"
mkdir -p experiments/logs/slurm

module load cuda/12.8.1

CONDA_BASE="/vast/parcc/spack/sw/apps/linux-sapphirerapids/anaconda3-2023.09-0-ieilyrkph5mewqcum3ajc4odlt2vakri"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "host=$(hostname)"
echo "python=$(which python)"
echo "python_version=$(python --version)"
echo "cuda=$(python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "sweep=$SWEEP count=$COUNT"

wandb agent --count "$COUNT" "$SWEEP"
