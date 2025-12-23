#!/bin/bash
#SBATCH --job-name=rkslsh
#SBATCH --partition=litian
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --array=0  # 5 models * 4 datasets * 4 budgets = 80 combos
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# Env
source /home/dixi/.cache/pypoetry/virtualenvs/kvpress-CimsZS3I-py3.10/bin/activate

# Huggingface
export HF_HOME=/net/projects2/litian-lab/dixi/cache/

export CUDA_LAUNCH_BLOCKING=1
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Paths
SCRIPT_PATH="reason/evaluate.py"
RESULT_DIR="reason/results"
mkdir -p logs "$RESULT_DIR"

# Sweep settings
PRESS_NAME="rkvlsh"
MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
  "meta-llama/Meta-Llama-3-8B"
)
DATASETS=(
  "gsm8k"
  "aime24"
  "aime25"
  "math500"
)
CACHE_BUDGETS=(128 256 384 512)
LAMBDA=0.01
N_HASH_BUCKETS=8

NUM_SAMPLES=30
RANDOM_SEED=42
MAX_NEW_TOKENS=2048

# Derived sizes
NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_BUDGETS=${#CACHE_BUDGETS[@]}
TOTAL=$((NUM_MODELS * NUM_DATASETS * NUM_BUDGETS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ $TASK_ID -ge $TOTAL ]]; then
  echo "TASK_ID $TASK_ID exceeds total jobs $TOTAL"
  exit 1
fi

# Map array index to (model, dataset, budget)
combo=$TASK_ID
model_idx=$(( combo / (NUM_DATASETS * NUM_BUDGETS) ))
rem=$(( combo % (NUM_DATASETS * NUM_BUDGETS) ))
dataset_idx=$(( rem / NUM_BUDGETS ))
budget_idx=$(( rem % NUM_BUDGETS ))

MODEL_NAME=${MODELS[$model_idx]}
DATASET=${DATASETS[$dataset_idx]}
CACHE_BUDGET=${CACHE_BUDGETS[$budget_idx]}
MODEL_FILE=${MODEL_NAME//\//--}

# Format lambda exactly like evaluate.py filenames
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

out_file="${RESULT_DIR}/${DATASET}____${MODEL_FILE}__${PRESS_NAME}__budget${CACHE_BUDGET}__hash_bucket${N_HASH_BUCKETS}__max_new_tokens${MAX_NEW_TOKENS}__lam${lambda_sanitized}__num_samples${NUM_SAMPLES}__sampling.jsonl"
score_file="${RESULT_DIR}/${DATASET}____${MODEL_FILE}__${PRESS_NAME}__budget${CACHE_BUDGET}__hash_bucket${N_HASH_BUCKETS}__max_new_tokens${MAX_NEW_TOKENS}__lam${lambda_sanitized}__num_samples${NUM_SAMPLES}__sampling_score.json"

if [[ -f "$score_file" ]]; then
  echo "✅ Skipping $DATASET @ budget $CACHE_BUDGET (score exists: $(basename "$score_file"))"
  exit 0
fi

if [[ -f "$out_file" ]]; then
  echo "⚠️  Results exist without score: $(basename "$out_file") — rerunning to generate score"
fi

echo "➡️  Running $DATASET | budget=$CACHE_BUDGET | lambda=$LAMBDA | model=$MODEL_NAME"
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
  --track_tokens=false \
  --measure_memory=false \
  --measure_latency=true

echo "✅ Done $DATASET | budget=$CACHE_BUDGET | lambda=$LAMBDA"