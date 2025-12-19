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

NUM_SAMPLES=15
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
CACHE_BUDGETS=(512) # 256 384 512
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
      # Format lambda to match evaluate.py exactly: multiply by 100, round, then format
      # e.g., 0 -> "0", 0.01 -> "001", 0.05 -> "005", 0.1 -> "01", 1.0 -> "1"
      # Use awk to round (matching Python's round() behavior)
      lambda_int=$(awk "BEGIN {printf \"%.0f\", $lambda * 100}")
      
      if [ "$lambda_int" -eq 0 ]; then
        lambda_sanitized="0"
      elif [ "$lambda_int" -lt 10 ]; then
        # 1-9: format as 3 digits with leading zeros (001, 002, ..., 009)
        lambda_sanitized=$(printf "%03d" "$lambda_int")
      elif [ "$lambda_int" -lt 100 ]; then
        # 10-99: format as 2 digits with leading zero (01, 02, ..., 09, 10, ..., 99)
        lambda_sanitized=$(printf "%02d" "$lambda_int")
      else
        # 100+: if divisible by 100, divide by 100, else keep as string
        if [ $((lambda_int % 100)) -eq 0 ]; then
          lambda_sanitized=$((lambda_int / 100))
        else
          lambda_sanitized="$lambda_int"
        fi
      fi
      
      # Construct filename matching evaluate.py format exactly
      out_file="${RESULT_DIR}/${dataset}____${MODEL_FILE}__${PRESS_NAME}__budget${budget}__hash_bucket8__max_new_tokens${MAX_NEW_TOKENS}__lam${lambda_sanitized}__num_samples${NUM_SAMPLES}__sampling.jsonl"
      
      # Also check for score file (evaluate.py checks score_filename for skip_existing)
      score_file="${RESULT_DIR}/${dataset}____${MODEL_FILE}__${PRESS_NAME}__budget${budget}__hash_bucket8__max_new_tokens${MAX_NEW_TOKENS}__lam${lambda_sanitized}__num_samples${NUM_SAMPLES}__sampling_score.json"
      
      if [[ -f "$out_file" ]] || [[ -f "$score_file" ]]; then
        existing_file=""
        if [[ -f "$score_file" ]]; then
          existing_file="score file: $(basename "$score_file")"
        elif [[ -f "$out_file" ]]; then
          existing_file="results file: $(basename "$out_file")"
        fi
        echo "    ✅ Skipping $dataset at budget $budget, lambda $lambda ($existing_file)"
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
        --lam="$lambda" \
        --track_tokens=False
    done
  done
done

echo "\n✅ All $PRESS_NAME evaluations complete."
