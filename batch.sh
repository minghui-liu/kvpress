nvidia-smi

export HF_HOME=../cache/
export HUGGINGFACE_TOKEN="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"
export CUDA_LAUNCH_BLOCKING=1
huggingface-cli login --token $HUGGINGFACE_TOKEN
set -euo pipefail

# ====== User-editable settings ======
# Set the press to evaluate (e.g. rkv, h2o, snapkv, knorm, full)
PRESS_NAME="rkv"

# Model and dataset settings
MODEL_NAME="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
# sanitize MODEL_NAME for filenames: replace "/" → "--"
MODEL_FILE="${MODEL_NAME//\//--}"

NUM_SAMPLES=100
RANDOM_SEED=42            # default seed in evaluate.py
MAX_NEW_TOKENS=2048

# Paths
SCRIPT_PATH="reason/evaluate.py"
RESULT_DIR="reason/results"

# ====== Benchmarks ======
datasets=(
  "gsm8k"               # openai/gsm8k
  # "math500"             # HuggingFaceH4/MATH-500
  #"commonsenseqa"       # tau/commonsense_qa
  #"openbookqa"          # allenai/openbookqa
  #"reclor"              # metaeval/reclor
  #"drop"                # ucinlp/drop
  #"strategyqa"          # ChilleD/StrategyQA
  #"folio"
  #  "aime25"
  #  "logiqa"
)

# Cache budgets to sweep over
CACHE_BUDGETS=(128 256 384 512)

# ====== Execution ======
echo "Starting $PRESS_NAME evaluations"
echo "Model: $MODEL_NAME | Samples: $NUM_SAMPLES | Seed: $RANDOM_SEED | Max new tokens: $MAX_NEW_TOKENS"

for budget in "${CACHE_BUDGETS[@]}"; do
  echo "\n🔄 Evaluating $PRESS_NAME with cache_budget=$budget..."
  for dataset in "${datasets[@]}"; do
    # Construct expected results filename
    out_file="${RESULT_DIR}/${dataset}____${MODEL_FILE}__${PRESS_NAME}__budget${budget}__max_new_tokens${MAX_NEW_TOKENS}__num_samples${NUM_SAMPLES}__sampling.jsonl"
    if [[ -f "$out_file" ]]; then
      echo "✅ Skipping $dataset at budget $budget (results exist: $(basename "$out_file"))"
      continue
    fi

    echo "➡️  Running $dataset @ budget $budget"
    CUDA_LAUNCH_BLOCKING=1 python "$SCRIPT_PATH" \
      --dataset="$dataset" \
      --model_name="$MODEL_NAME" \
      --press_name="$PRESS_NAME" \
      --cache_budget="$budget" \
      --num_samples="$NUM_SAMPLES" \
      --random_seed="$RANDOM_SEED" \
      --max_new_tokens="$MAX_NEW_TOKENS" \
      --n_hash_buckets=4 
  done
done

echo "\n✅ All $PRESS_NAME evaluations complete."
