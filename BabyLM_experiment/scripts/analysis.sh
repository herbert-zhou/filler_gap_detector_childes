#!/bin/bash

#SBATCH --time 0:08:00
#SBATCH --gpus=a100:1
#SBATCH --mem 32G
#SBATCH --partition gpu
#SBATCH --array=0-959
#SBATCH --output=../slurm_logs/analysis_%A_%a.out
#SBATCH --error=../slurm_logs/analysis_%A_%a.err

# Define parameter arrays
DATASET_FILTERS=("embedded_questions" "matrix_questions" "relative_clauses")
MODEL_CONFIGS=("gpt-705M")
DATASET_SIZES=("10M")
INCLUDE_DIRS=("par")
SEEDS=(0 1 2 3 4)
CONTROL_FLAGS=(0 1)  # 0 = filtered, 1 = control
LEARNING_RATES=(3e-4)  # Default learning rate
WEIGHT_DECAYS=(0.0)    # Default weight decay

# All minimal pair categories from sentence_dict in analysis.py
MINIMAL_PAIR_CATEGORIES=(
    "nogap_animate_embedded"
    "gap_animate_embedded"
    "nogap_animate_matrix"
    "gap_animate_matrix"
    "nogap_inanimate_matrix"
    "gap_inanimate_matrix"
    "nogap_inanimate_embedded"
    "gap_inanimate_embedded"
    "intransitive"
    "gap_subj_embedded"
    "nogap_subj_embedded"
    "gap_subj_matrix"
    "nogap_subj_matrix"
    "gap_relative"
    "nogap_relative"
    "animate_embedded"
    "inanimate_embedded"
    "animate_matrix"
    "inanimate_matrix"
    "subj_embedded"
    "subj_matrix"
    "embedded"
    "matrix"
    "relative"
    "animate_matrix_strict"
    "inanimate_matrix_strict"
    "matrix_strict"
    "animate_matrix_qmark"
    "animate_matrix_period"
    "inanimate_matrix_qmark"
    "inanimate_matrix_period"
    "matrix_continuation"
)

# Chunking support: Set CHUNK_START and CHUNK_END environment variables to run subset of jobs
# Example: CHUNK_START=0 CHUNK_END=100 sbatch analysis.sh
# If not set, run all jobs
CHUNK_START=${CHUNK_START:-0}
CHUNK_END=${CHUNK_END:-959}

# Check if this task should be skipped based on chunking
if [ $SLURM_ARRAY_TASK_ID -lt $CHUNK_START ] || [ $SLURM_ARRAY_TASK_ID -gt $CHUNK_END ]; then
    echo "Skipping task $SLURM_ARRAY_TASK_ID (outside chunk range $CHUNK_START-$CHUNK_END)"
    exit 0
fi

# Calculate indices from SLURM_ARRAY_TASK_ID
# Total combinations: 3 filters * 1 model * 1 size * 1 include * 5 seeds * 2 control * 32 categories = 960
NUM_FILTERS=${#DATASET_FILTERS[@]}
NUM_MODELS=${#MODEL_CONFIGS[@]}
NUM_SIZES=${#DATASET_SIZES[@]}
NUM_INCLUDES=${#INCLUDE_DIRS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONTROL=${#CONTROL_FLAGS[@]}
NUM_CATEGORIES=${#MINIMAL_PAIR_CATEGORIES[@]}

# Decompose task ID into indices
task_id=$SLURM_ARRAY_TASK_ID
category_idx=$((task_id % NUM_CATEGORIES))
task_id=$((task_id / NUM_CATEGORIES))
control_idx=$((task_id % NUM_CONTROL))
task_id=$((task_id / NUM_CONTROL))
seed_idx=$((task_id % NUM_SEEDS))
task_id=$((task_id / NUM_SEEDS))
include_idx=$((task_id % NUM_INCLUDES))
task_id=$((task_id / NUM_INCLUDES))
size_idx=$((task_id % NUM_SIZES))
task_id=$((task_id / NUM_SIZES))
model_idx=$((task_id % NUM_MODELS))
task_id=$((task_id / NUM_MODELS))
filter_idx=$((task_id % NUM_FILTERS))

# Get parameter values
DATASET_FILTER=${DATASET_FILTERS[$filter_idx]}
MODEL_CONFIG=${MODEL_CONFIGS[$model_idx]}
DATASET_SIZE=${DATASET_SIZES[$size_idx]}
INCLUDE_DIR=${INCLUDE_DIRS[$include_idx]}
SEED=${SEEDS[$seed_idx]}
IS_CONTROL=${CONTROL_FLAGS[$control_idx]}
MINIMAL_PAIR_CATEGORY=${MINIMAL_PAIR_CATEGORIES[$category_idx]}

# Set control flag for python script
if [ $IS_CONTROL -eq 1 ]; then
    CONTROL_ARG="--control"
    MODEL_TYPE="control"
else
    CONTROL_ARG=""
    MODEL_TYPE="filtered"
fi

# Get learning rate and weight decay (using first element since we only have one value for now)
LR=${LEARNING_RATES[0]}
WD=${WEIGHT_DECAYS[0]}

echo "Running analysis job $SLURM_ARRAY_TASK_ID: filter=$DATASET_FILTER, model=$MODEL_CONFIG, size=$DATASET_SIZE, include=$INCLUDE_DIR, seed=$SEED, lr=$LR, wd=$WD, type=$MODEL_TYPE, category=$MINIMAL_PAIR_CATEGORY"

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

python analysis.py \
    --dataset_filter $DATASET_FILTER \
    --model_config $MODEL_CONFIG \
    --dataset_size $DATASET_SIZE \
    --include_dir $INCLUDE_DIR \
    --seed $SEED \
    --minimal_pair_category $MINIMAL_PAIR_CATEGORY \
    --lr $LR \
    --wd $WD \
    $CONTROL_ARG