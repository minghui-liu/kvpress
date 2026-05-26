#!/bin/bash
#SBATCH --job-name=seerattention
#SBATCH --partition=litian,general
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-239
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.txt

set -euo pipefail

# Env
source /home/dixi/.cache/pypoetry/virtualenvs/kvpress-CimsZS3I-py3.10/bin/activate

# Hugging Face/cache
export HF_HOME=/net/projects2/litian-lab/dixi/cache/
export CUDA_LAUNCH_BLOCKING=1

# Paths
SCRIPT_PATH="reason/evaluate.py"
RESULT_DIR="reason/results"
mkdir -p logs "$RESULT_DIR"

# SeerAttention settings
MODEL_NAME="SeerAttention/SeerAttention-Decode-R1-Distill-Qwen-14B-AttnGates"
PRESS_METHOD="none"
CACHE_BUDGETS=(128 256 384 512)

# Dataset/settings sweep
DATASETS=("gsm8k" "math500")
NUM_SAMPLES=100
RANDOM_SEEDS=(24 42 130)
BLOCK_SIZE=q0
BLOCK_INDICES=(1 2 3 4 5 6 7 8 9 10)
TRACK_TOKENS=false

# =====================
# Derived sizes
# =====================
NUM_DATASETS=${#DATASETS[@]}
NUM_BUDGETS=${#CACHE_BUDGETS[@]}
NUM_SEEDS=${#RANDOM_SEEDS[@]}
NUM_BLOCKS=${#BLOCK_INDICES[@]}
TOTAL=$((NUM_DATASETS * NUM_BUDGETS * NUM_SEEDS * NUM_BLOCKS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ $TASK_ID -ge $TOTAL ]]; then
  echo "TASK_ID $TASK_ID exceeds total jobs $TOTAL"
  exit 1
fi

# Map array index to (dataset, budget, seed, block)
combo=$TASK_ID
dataset_idx=$(( combo / (NUM_BUDGETS * NUM_SEEDS * NUM_BLOCKS) ))
rem=$(( combo % (NUM_BUDGETS * NUM_SEEDS * NUM_BLOCKS) ))
budget_idx=$(( rem / (NUM_SEEDS * NUM_BLOCKS) ))
rem=$(( rem % (NUM_SEEDS * NUM_BLOCKS) ))
seed_idx=$(( rem / NUM_BLOCKS ))
block_idx=$(( rem % NUM_BLOCKS ))

DATASET=${DATASETS[$dataset_idx]}
CACHE_BUDGET=${CACHE_BUDGETS[$budget_idx]}
RANDOM_SEED=${RANDOM_SEEDS[$seed_idx]}
DATASET_BLOCK_INDEX=${BLOCK_INDICES[$block_idx]}
MODEL_FILE=${MODEL_NAME//\//--}

# =====================
# Dataset-specific max tokens
# =====================
case "$DATASET" in
  aime24)
    MAX_NEW_TOKENS=32768
    ;;
  aime25)
    MAX_NEW_TOKENS=32768
    ;;
  math500)
    MAX_NEW_TOKENS=16384
    ;;
  gsm8k)
    MAX_NEW_TOKENS=5096
    ;;
  drop)
    MAX_NEW_TOKENS=5096
    ;;
  reclor)
    MAX_NEW_TOKENS=5096
    ;;
  folio)
    MAX_NEW_TOKENS=5096
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

file_stem="${DATASET}____${MODEL_FILE}__${PRESS_METHOD}__budget${CACHE_BUDGET}__max_new_tokens${MAX_NEW_TOKENS}__num_samples${NUM_SAMPLES}__block${DATASET_BLOCK_INDEX}_size${BLOCK_SIZE}__seed${RANDOM_SEED}__sampling"
out_file="${RESULT_DIR}/${file_stem}.jsonl"
score_file="${RESULT_DIR}/${file_stem}_score.json"

if [[ -f "$score_file" ]]; then
  echo "Skipping $DATASET | budget=$CACHE_BUDGET | seed=$RANDOM_SEED | block=$DATASET_BLOCK_INDEX (score exists: $(basename "$score_file"))"
  exit 0
fi

if [[ -f "$out_file" ]]; then
  echo "Results exist without score: $(basename "$out_file"); rerunning to generate score"
fi

echo "Running SeerAttention | dataset=$DATASET | budget=$CACHE_BUDGET | seed=$RANDOM_SEED | block=$DATASET_BLOCK_INDEX/${NUM_BLOCKS} | model=$MODEL_NAME"
python "$SCRIPT_PATH" \
  --dataset="$DATASET" \
  --model_name="$MODEL_NAME" \
  --press_name="$PRESS_METHOD" \
  --cache_budget="$CACHE_BUDGET" \
  --num_samples="$NUM_SAMPLES" \
  --dataset_block_index="$DATASET_BLOCK_INDEX" \
  --dataset_block_size="$BLOCK_SIZE" \
  --random_seed="$RANDOM_SEED" \
  --max_new_tokens="$MAX_NEW_TOKENS" \
  --track_tokens="$TRACK_TOKENS" \
  --measure_memory=false \
  --measure_latency=false \
  --temperature=0.6

echo "Done SeerAttention | dataset=$DATASET | budget=$CACHE_BUDGET | seed=$RANDOM_SEED | block=$DATASET_BLOCK_INDEX"
