#!/bin/bash

#SBATCH --time 0:30:00
#SBATCH --mem 64G
#SBATCH --partition day
#SBATCH --array=0-35
#SBATCH --output=../slurm_logs/data_processor_%A_%a.out
#SBATCH --error=../slurm_logs/data_processor_%A_%a.err

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

filters=(unfiltered matrix_questions embedded_questions relative_clauses)
splits=(dev test train_10M train_100M_part1 train_100M_part2 train_100M_part3 train_100M_part4 train_100M_part5 train_100M_part6)

num_splits=${#splits[@]}
filter_index=$((SLURM_ARRAY_TASK_ID / num_splits))
split_index=$((SLURM_ARRAY_TASK_ID % num_splits))

sentence_filter=${filters[$filter_index]}
split=${splits[$split_index]}

echo "Processing filter=${sentence_filter} split=${split}"

python data_processor.py --sentence_filter "$sentence_filter" --split "$split"