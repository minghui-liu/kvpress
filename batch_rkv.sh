#!/bin/bash
#SBATCH --job-name=rkvlsh_qual
#SBATCH --partition=litian,general
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=2-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# =====================
# Environment
# =====================
source /home/dixi/.cache/pypoetry/virtualenvs/kvpress-CimsZS3I-py3.10/bin/activate

# =====================
# Hugging Face / CUDA
# =====================
export HF_HOME=/net/projects2/litian-lab/dixi/cache/
export CUDA_LAUNCH_BLOCKING=1

# =====================
# Paths
# =====================
SCRIPT_PATH="reason/evaluate.py"
RESULT_DIR="reason/results"
mkdir -p logs "$RESULT_DIR"

# =====================
# Sweep settings
# =====================
# Test both RKV and RKV-LSH
PRESS_NAMES=("rkv" "rkvlsh")

MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)

DATASETS=(
  "math500"
)

CACHE_BUDGETS=(1024)
LAMBDA=0.1
N_HASH_BUCKETS=8

NUM_SAMPLES=15
RANDOM_SEED=42

# Max tokens modes to traverse
MAX_TOKENS_MODES=("separate")

# =====================
# Derived sizes
# =====================
NUM_PRESS=${#PRESS_NAMES[@]}
NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_BUDGETS=${#CACHE_BUDGETS[@]}
NUM_MODES=${#MAX_TOKENS_MODES[@]}
JOBS_PER_MODE=$((NUM_PRESS * NUM_MODELS * NUM_DATASETS * NUM_BUDGETS))
TOTAL=$((NUM_MODES * JOBS_PER_MODE))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ $TASK_ID -ge $TOTAL ]]; then
  echo "TASK_ID $TASK_ID exceeds total jobs $TOTAL"
  exit 1
fi

# =====================
# Map array index → (mode, press, model, dataset, budget)
# =====================
mode_idx=$(( TASK_ID / JOBS_PER_MODE ))
combo=$(( TASK_ID % JOBS_PER_MODE ))
MAX_TOKENS_MODE=${MAX_TOKENS_MODES[$mode_idx]}

press_idx=$(( combo / (NUM_MODELS * NUM_DATASETS * NUM_BUDGETS) ))
rem=$(( combo % (NUM_MODELS * NUM_DATASETS * NUM_BUDGETS) ))
model_idx=$(( rem / (NUM_DATASETS * NUM_BUDGETS) ))
rem2=$(( rem % (NUM_DATASETS * NUM_BUDGETS) ))
dataset_idx=$(( rem2 / NUM_BUDGETS ))
budget_idx=$(( rem2 % NUM_BUDGETS ))

PRESS_NAME=${PRESS_NAMES[$press_idx]}
MODEL_NAME=${MODELS[$model_idx]}
DATASET=${DATASETS[$dataset_idx]}
CACHE_BUDGET=${CACHE_BUDGETS[$budget_idx]}
MODEL_FILE=${MODEL_NAME//\//--}

# Resolve max tokens per mode/dataset
resolve_max_tokens() {
  local dataset="$1"
  case "$MAX_TOKENS_MODE" in
    force2048)
      echo 2048
      ;;
    separate)
      case "$dataset" in
        math500) echo "16384" ;;
        aime24)  echo "32768" ;;
        *)       echo "2048"  ;;
      esac
      ;;
    *)
      case "$dataset" in
        aime24)
          echo "32768"
          ;;
        math500)
          echo "16384"
          ;;
        *)
          echo "2048"
          ;;
      esac
      ;;
  esac
}

MAX_NEW_TOKENS=$(resolve_max_tokens "$DATASET")

# =====================
# Lambda formatting
# =====================
lambda_int=$(awk "BEGIN {printf \"%.0f\", $LAMBDA * 100}")
if [ "$lambda_int" -eq 0 ]; then
  lambda_sanitized="0"
elif [ "$lambda_int" -lt 10 ]; then
  lambda_sanitized=$(printf "%03d" "$lambda_int")
elif [ "$lambda_int" -lt 100 ]; then
  lambda_sanitized=$(printf "%02d" "$lambda_int")
else
  if [ $((lambda_int % 100)) -eq 0 ]; then
    lambda_sanitized=$((lambda_int / 100))
  else
    lambda_sanitized="$lambda_int"
  fi
fi

# =====================
# Output files
# =====================
out_file="${RESULT_DIR}/${DATASET}____${MODEL_FILE}__${PRESS_NAME}__budget${CACHE_BUDGET}__hash_bucket${N_HASH_BUCKETS}__max_new_tokens${MAX_NEW_TOKENS}__lam${lambda_sanitized}__num_samples${NUM_SAMPLES}__sampling.jsonl"
score_file="${RESULT_DIR}/${DATASET}____${MODEL_FILE}__${PRESS_NAME}__budget${CACHE_BUDGET}__hash_bucket${N_HASH_BUCKETS}__max_new_tokens${MAX_NEW_TOKENS}__lam${lambda_sanitized}__num_samples${NUM_SAMPLES}__sampling_score.json"

# =====================
# Skip logic
# =====================
if [[ -f "$score_file" ]]; then
  echo "Skipping $DATASET @ press=$PRESS_NAME @ budget $CACHE_BUDGET (score exists)"
  exit 0
fi

if [[ -f "$out_file" ]]; then
  echo "Results exist without score — rerunning to generate score"
fi

# =====================
# Run
# =====================
echo "Running dataset=$DATASET | model=$MODEL_NAME | press=$PRESS_NAME | budget=$CACHE_BUDGET | max_new_tokens=$MAX_NEW_TOKENS | max_tokens_mode=$MAX_TOKENS_MODE | lambda=$LAMBDA"

python "$SCRIPT_PATH" \
  --dataset="$DATASET" \
  --model_name="$MODEL_NAME" \
  --press_name="$PRESS_NAME" \
  --cache_budget="$CACHE_BUDGET" \
  --num_samples="$NUM_SAMPLES" \
  --random_seed="$RANDOM_SEED" \
  --max_new_tokens="$MAX_NEW_TOKENS" \
  --n_hash_buckets="$N_HASH_BUCKETS" \
  --lam="$LAMBDA" \
  --track_tokens=true \
  --enable_qualitative_analysis=true \
  --measure_memory=false \
  --measure_latency=true

echo "Done $DATASET | press=$PRESS_NAME | budget=$CACHE_BUDGET | lambda=$LAMBDA"
