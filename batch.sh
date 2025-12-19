nvidia-smi

export HF_HOME=../cache/
# export HUGGINGFACE_TOKEN="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"
export CUDA_LAUNCH_BLOCKING=1
huggingface-cli login --token $HUGGINGFACE_TOKEN
set -euo pipefail

# ====== User-editable settings ======
# Set the press to evaluate (e.g. rkv, h2o, snapkv, knorm, full)
PRESS_NAME="rkvlsh"

# Model and dataset settings
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# sanitize MODEL_NAME for filenames: replace "/" → "--"
MODEL_FILE="${MODEL_NAME//\//--}"

NUM_SAMPLES=10
RANDOM_SEED=42            # default seed in evaluate.py
MAX_NEW_TOKENS=16384

# Paths
SCRIPT_PATH="reason/evaluate.py"
RESULT_DIR="reason/results"

# ====== Benchmarks ======
datasets=(
  # "gsm8k"               # openai/gsm8k
  "math500"             # HuggingFaceH4/MATH-500
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
CACHE_BUDGETS=(128) # 256 384 512
LAMBS=(0 0.01 0.05 0.1 0.2 0.4 1)

# ====== Execution ======
echo "Starting $PRESS_NAME evaluations"
echo "Model: $MODEL_NAME | Samples: $NUM_SAMPLES | Seed: $RANDOM_SEED | Max new tokens: $MAX_NEW_TOKENS"
echo "Lambda values to test: ${LAMBS[*]}"

for budget in "${CACHE_BUDGETS[@]}"; do
  echo "\n🔄 Evaluating $PRESS_NAME with cache_budget=$budget..."
  for lambda in "${LAMBS[@]}"; do
    echo "\n  📊 Testing lambda=$lambda..."
    for dataset in "${datasets[@]}"; do
      # Construct expected results filename (include lambda in filename)
      # Sanitize lambda for filename: replace "." with "p" (e.g., 0.1 -> 0p1)
      lambda_sanitized="${lambda//./p}"
      out_file="${RESULT_DIR}/${dataset}____${MODEL_FILE}__${PRESS_NAME}__budget${budget}__lam${lambda_sanitized}__max_new_tokens${MAX_NEW_TOKENS}__num_samples${NUM_SAMPLES}__sampling.jsonl"
      if [[ -f "$out_file" ]]; then
        echo "    ✅ Skipping $dataset at budget $budget, lambda $lambda (results exist: $(basename "$out_file"))"
        continue
      fi

      echo "    ➡️  Running $dataset @ budget $budget, lambda $lambda"
      CUDA_LAUNCH_BLOCKING=1 python "$SCRIPT_PATH" \
        --dataset="$dataset" \
        --model_name="$MODEL_NAME" \
        --press_name="$PRESS_NAME" \
        --cache_budget="$budget" \
        --num_samples="$NUM_SAMPLES" \
        --random_seed="$RANDOM_SEED" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --n_hash_buckets=8 \
        --lam="$lambda"
    done
  done
done

echo "\n✅ All $PRESS_NAME evaluations complete."
