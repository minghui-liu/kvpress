#!/bin/bash
#SBATCH --job-name=semantic
#SBATCH --partition=litian,general
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-399
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.txt

set -euo pipefail

# Env
# conda init
# conda activate py310
source /home/dixi/.cache/pypoetry/virtualenvs/kvpress-CimsZS3I-py3.10/bin/activate

# Huggingface
export HF_HOME=/net/projects2/litian-lab/dixi/cache/
export CUDA_LAUNCH_BLOCKING=1

# Paths
SCRIPT_PATH="semantic.py"
RESULTS_DIR="${RESULTS_DIR:-reason/results}"
OUTPUT_DIR="${OUTPUT_DIR:-semantic_results}"
mkdir -p logs "$OUTPUT_DIR"

# Sweep settings
METHODS=("rkv" "h2o" "knorm" "snapkv" "streaming_llm")
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"  # ML
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # DQ
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"  # LN
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # DL
    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # DQ
)
DATASETS=("gsm8k" "math500" "folio" "reclor" "drop")
BUDGETS=(128 256 384 512)

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

# =====================
# Derived sizes
# =====================
NUM_METHODS=${#METHODS[@]}
NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_BUDGETS=${#BUDGETS[@]}
TOTAL=$((NUM_METHODS * NUM_MODELS * NUM_DATASETS * NUM_BUDGETS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ $TASK_ID -ge $TOTAL ]]; then
  echo "TASK_ID $TASK_ID exceeds total jobs $TOTAL"
  exit 1
fi

# Map array index to (method, model, dataset, budget)
combo=$TASK_ID
method_idx=$(( combo / (NUM_MODELS * NUM_DATASETS * NUM_BUDGETS) ))
rem=$(( combo % (NUM_MODELS * NUM_DATASETS * NUM_BUDGETS) ))
model_idx=$(( rem / (NUM_DATASETS * NUM_BUDGETS) ))
rem=$(( rem % (NUM_DATASETS * NUM_BUDGETS) ))
dataset_idx=$(( rem / NUM_BUDGETS ))
budget_idx=$(( rem % NUM_BUDGETS ))

METHOD=${METHODS[$method_idx]}
MODEL_NAME=${MODELS[$model_idx]}
DATASET=${DATASETS[$dataset_idx]}
BUDGET=${BUDGETS[$budget_idx]}
MODEL_FILE=${MODEL_NAME//\//--}

out_file="${OUTPUT_DIR}/${DATASET}__${MODEL_FILE}__${METHOD}__budget${BUDGET}_semantic.json"

if [[ -f "$out_file" ]]; then
  echo "Skipping semantic result exists: $(basename "$out_file")"
  exit 0
fi

DEVICE_ARG=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARG=(--device "$DEVICE")
fi

echo "Running semantic similarity | dataset=$DATASET | model=$MODEL_NAME | method=$METHOD | budget=$BUDGET | task=$TASK_ID/$((TOTAL - 1))"
python "$SCRIPT_PATH" \
  --results_dir "$RESULTS_DIR" \
  --dataset "$DATASET" \
  --model_name "$MODEL_NAME" \
  --method_name "$METHOD" \
  --budget "$BUDGET" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --output "$out_file" \
  "${DEVICE_ARG[@]}"

echo "Done semantic similarity | dataset=$DATASET | model=$MODEL_NAME | method=$METHOD | budget=$BUDGET"
