#!/bin/bash

#SBATCH --time 0:25:00
#SBATCH --gpus=a100:1
#SBATCH --mem 32G
#SBATCH --partition gpu
#SBATCH --array=0-104
#SBATCH --job-name=gpt_10M
#SBATCH --output=../slurm_logs/gpt_10M_%A_%a.out
#SBATCH --error=../slurm_logs/gpt_10M_%A_%a.err

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ling-gen

# Resolve project root robustly for SLURM (scripts may run from /var/spool/slurmd)
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/../src" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT/src"

# Define all model configurations (filter, is_control)
# Jobs 0-2: Control models for each filtered condition
# Jobs 3-5: Filtered models for each filtered condition
# Job 6: Unfiltered model
configs=(
    "embedded_questions:1"
    "matrix_questions:1"
    "relative_clauses:1"
    "embedded_questions:0"
    "matrix_questions:0"
    "relative_clauses:0"
    "unfiltered:0"
)

size="10M"
model="gpt-705M"
include="par"
num_seeds=15

# Calculate which config and seed this job should run
config_idx=$((SLURM_ARRAY_TASK_ID / num_seeds))
seed=$((SLURM_ARRAY_TASK_ID % num_seeds))

# Get configuration for this array task
IFS=':' read -r filter is_control <<< "${configs[$config_idx]}"

if [ "$is_control" -eq 1 ]; then
    echo "Running control for filter: $filter, seed: $seed (task $SLURM_ARRAY_TASK_ID)"
    python train.py --config ../config/$model.yaml --filter $filter --size $size --include $include --seed $seed --control
else
    echo "Running filter: $filter, seed: $seed (task $SLURM_ARRAY_TASK_ID)"
    python train.py --config ../config/$model.yaml --filter $filter --size $size --include $include --seed $seed
fi