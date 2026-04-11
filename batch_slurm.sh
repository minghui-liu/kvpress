#!/bin/bash
#SBATCH --job-name=dpbench
#SBATCH --partition=litian,general
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-2519
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.txt

set -euo pipefail

# Env
source /home/dixi/.cache/pypoetry/virtualenvs/kvpress-CimsZS3I-py3.10/bin/activate

# Huggingface
export HF_HOME=/net/projects2/litian-lab/dixi/cache/
export CUDA_LAUNCH_BLOCKING=1

# Paths
SCRIPT_PATH="reason/evaluate.py"
RESULT_DIR="reason/results"
mkdir -p logs "$RESULT_DIR"

# Sweep settings
PRESS_NAME=("rkv" "h2o" "knorm" "snapkv" "streaming_llm" "full")
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"  # ML
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # DQ
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"  # LN
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # DL
)
DATASETS=(
  "math500"
)
CACHE_BUDGETS=(128 256 384 512)
LAMBDA=0
N_HASH_BUCKETS=8

NUM_SAMPLES=500
RANDOM_SEEDS=(24 42 130)
BLOCK_SIZE=50
BLOCK_INDICES=(1 2 3 4 5 6 7 8 9 10)

# =====================
# Derived sizes
# =====================
NUM_SEEDS=${#RANDOM_SEEDS[@]}
NUM_BLOCKS=${#BLOCK_INDICES[@]}

SPEC_MODELS=()
SPEC_DATASETS=()
SPEC_PRESSES=()
SPEC_BUDGETS=()

for MODEL_NAME in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRESS_METHOD in "${PRESS_NAME[@]}"; do
      if [[ "$PRESS_METHOD" == "full" ]]; then
        SPEC_MODELS+=("$MODEL_NAME")
        SPEC_DATASETS+=("$DATASET")
        SPEC_PRESSES+=("$PRESS_METHOD")
        SPEC_BUDGETS+=("${CACHE_BUDGETS[0]}")
      else
        for CACHE_BUDGET in "${CACHE_BUDGETS[@]}"; do
          SPEC_MODELS+=("$MODEL_NAME")
          SPEC_DATASETS+=("$DATASET")
          SPEC_PRESSES+=("$PRESS_METHOD")
          SPEC_BUDGETS+=("$CACHE_BUDGET")
        done
      fi
    done
  done
done

NUM_SPECS=${#SPEC_MODELS[@]}
TOTAL=$((NUM_SPECS * NUM_SEEDS * NUM_BLOCKS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ $TASK_ID -ge $TOTAL ]]; then
  echo "TASK_ID $TASK_ID exceeds total jobs $TOTAL"
  exit 1
fi

# Map array index to (model, dataset, press_method, budget, seed, block)
combo=$TASK_ID
spec_idx=$(( combo / (NUM_SEEDS * NUM_BLOCKS) ))
rem=$(( combo % (NUM_SEEDS * NUM_BLOCKS) ))
seed_idx=$(( rem / NUM_BLOCKS ))
block_idx=$(( rem % NUM_BLOCKS ))

MODEL_NAME=${SPEC_MODELS[$spec_idx]}
DATASET=${SPEC_DATASETS[$spec_idx]}
PRESS_METHOD=${SPEC_PRESSES[$spec_idx]}
CACHE_BUDGET=${SPEC_BUDGETS[$spec_idx]}
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
  math500)
    MAX_NEW_TOKENS=16384
    ;;
  gsm8k)
    MAX_NEW_TOKENS=16384
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

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

if [[ "$PRESS_METHOD" == "rkv" || "$PRESS_METHOD" == "rkvlsh" ]]; then
  file_stem="${DATASET}____${MODEL_FILE}__${PRESS_METHOD}__budget${CACHE_BUDGET}__hash_bucket${N_HASH_BUCKETS}__max_new_tokens${MAX_NEW_TOKENS}__lam${lambda_sanitized}__num_samples${NUM_SAMPLES}__block${DATASET_BLOCK_INDEX}_size${BLOCK_SIZE}__seed${RANDOM_SEED}__sampling"
elif [[ "$PRESS_METHOD" == "turboquant" ]]; then
  file_stem="${DATASET}____${MODEL_FILE}__${PRESS_METHOD}__int${N_BITS}__max_new_tokens${MAX_NEW_TOKENS}__num_samples${NUM_SAMPLES}__block${DATASET_BLOCK_INDEX}_size${BLOCK_SIZE}__seed${RANDOM_SEED}__sampling"
else
  file_stem="${DATASET}____${MODEL_FILE}__${PRESS_METHOD}__budget${CACHE_BUDGET}__max_new_tokens${MAX_NEW_TOKENS}__num_samples${NUM_SAMPLES}__block${DATASET_BLOCK_INDEX}_size${BLOCK_SIZE}__seed${RANDOM_SEED}__sampling"
fi

out_file="${RESULT_DIR}/${file_stem}.jsonl"
score_file="${RESULT_DIR}/${file_stem}_score.json"

if [[ -f "$score_file" ]]; then
  echo "✅ Skipping $DATASET @ press=$PRESS_METHOD @ budget $CACHE_BUDGET @ block=$DATASET_BLOCK_INDEX (score exists: $(basename "$score_file"))"
  exit 0
fi

if [[ -f "$out_file" ]]; then
  echo "⚠️  Results exist without score: $(basename "$out_file") — rerunning to generate score"
fi

echo "➡️  Running $DATASET | press=$PRESS_METHOD | budget=$CACHE_BUDGET | lambda=$LAMBDA | seed=$RANDOM_SEED | block=$DATASET_BLOCK_INDEX/${NUM_BLOCKS} | model=$MODEL_NAME"
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
  --n_hash_buckets="$N_HASH_BUCKETS" \
  --lam="$LAMBDA" \
  --track_tokens=false \
  --measure_memory=false \
  --measure_latency=false

echo "✅ Done $DATASET | press=$PRESS_METHOD | budget=$CACHE_BUDGET | lambda=$LAMBDA | block=$DATASET_BLOCK_INDEX"
