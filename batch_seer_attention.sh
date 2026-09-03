#!/bin/bash
# Batch script for SeerAttention model evaluation across cache budgets

set -euo pipefail

export HF_HOME=../cache/
# export HUGGINGFACE_TOKEN="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"
export CUDA_LAUNCH_BLOCKING=1
huggingface-cli login --token $HUGGINGFACE_TOKEN

# ====== Configuration ======
# Task range: set to "0-4" for first 5 tasks, "5-9" for tasks 6-10, etc.
# Leave empty or unset to run all tasks
TASK_RANGE="${TASK_RANGE:-}"

# Number of parallel jobs (set to number of GPUs or desired parallelism)
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Token tracking: set to "true" to track retained/evicted tokens, "false" to skip (faster)
TRACK_TOKENS="${TRACK_TOKENS:-false}"

NUM_SAMPLES=100
RANDOM_SEED=42
MAX_NEW_TOKENS=2048
DATASET="gsm8k"

# Model to test
MODEL="SeerAttention/SeerAttention-Decode-R1-Distill-Qwen-14B-AttnGates"

# Press method (use "full" for no compression)
PRESS="none"

# Cache budgets to test (note: 4096 first, then others)
CACHE_BUDGETS=(384 512 4096)

# Paths
SCRIPT_PATH="reason/evaluate.py"
RESULT_DIR="reason/results"

# ====== Helper Functions ======
# Parse task range (e.g., "0-4" -> start=0, end=4)
parse_task_range() {
    if [[ -z "$TASK_RANGE" ]]; then
        TASK_START=0
        TASK_END=999999
        return
    fi
    
    if [[ "$TASK_RANGE" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        TASK_START="${BASH_REMATCH[1]}"
        TASK_END="${BASH_REMATCH[2]}"
    else
        echo "Error: TASK_RANGE must be in format 'start-end' (e.g., '0-4')"
        exit 1
    fi
}

# Check if task index is in range
is_task_in_range() {
    local task_idx=$1
    if [[ $task_idx -ge $TASK_START && $task_idx -le $TASK_END ]]; then
        return 0
    else
        return 1
    fi
}

# Run a single task
run_task() {
    local task_idx=$1
    local model_name=$2
    local press_name=$3
    local budget=$4
    
    MODEL_FILE="${model_name//\//--}"
    
    # Construct expected results filename
    out_file="${RESULT_DIR}/${DATASET}____${MODEL_FILE}__${press_name}__budget${budget}__max_new_tokens${MAX_NEW_TOKENS}__num_samples${NUM_SAMPLES}__sampling.jsonl"
    
    if [[ -f "$out_file" ]]; then
        echo "[Task $task_idx] ✅ Skipping $model_name | $press_name | budget=$budget (results exist)"
        return 0
    fi
    
    echo "[Task $task_idx] 🔄 Starting $model_name | $press_name | budget=$budget"
    
    # Build command
    local cmd="python $SCRIPT_PATH"
    cmd="$cmd --dataset=$DATASET"
    cmd="$cmd --model_name=\"$model_name\""
    cmd="$cmd --press_name=$press_name"
    cmd="$cmd --cache_budget=$budget"
    cmd="$cmd --num_samples=$NUM_SAMPLES"
    cmd="$cmd --random_seed=$RANDOM_SEED"
    cmd="$cmd --max_new_tokens=$MAX_NEW_TOKENS"
    cmd="$cmd --track_tokens=$TRACK_TOKENS"
    
    # Run the command
    if eval $cmd; then
        echo "[Task $task_idx] ✅ Completed $model_name | $press_name | budget=$budget"
        return 0
    else
        echo "[Task $task_idx] ❌ Failed $model_name | $press_name | budget=$budget"
        return 1
    fi
}

# ====== Execution ======
parse_task_range

echo "Starting SeerAttention model evaluation"
echo "Dataset: $DATASET | Samples: $NUM_SAMPLES | Seed: $RANDOM_SEED | Max new tokens: $MAX_NEW_TOKENS"
echo "Model: $MODEL"
echo "Press: $PRESS"
echo "Cache budgets: ${CACHE_BUDGETS[@]}"
echo "Track tokens: $TRACK_TOKENS"
if [[ -n "$TASK_RANGE" ]]; then
    echo "Task range: $TASK_RANGE (tasks $TASK_START to $TASK_END)"
else
    echo "Task range: ALL (running all tasks)"
fi
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

# Collect all tasks
declare -a tasks
task_idx=0

for budget in "${CACHE_BUDGETS[@]}"; do
    if is_task_in_range $task_idx; then
        tasks+=("$task_idx|$MODEL|$PRESS|$budget")
    fi
    task_idx=$((task_idx + 1))
done

total_tasks=${#tasks[@]}
echo "Total tasks to run: $total_tasks"
echo ""

# Function to run tasks with parallel job control
run_parallel_tasks() {
    local running=0
    local completed=0
    local failed=0
    declare -a pids
    
    for task_str in "${tasks[@]}"; do
        IFS='|' read -r idx model press budget <<< "$task_str"
        
        # Wait for a slot if we're at max parallel jobs
        while [[ $running -ge $PARALLEL_JOBS ]]; do
            for pid in "${!pids[@]}"; do
                if ! kill -0 "${pids[$pid]}" 2>/dev/null; then
                    # Process finished
                    wait "${pids[$pid]}"
                    exit_code=$?
                    if [[ $exit_code -eq 0 ]]; then
                        completed=$((completed + 1))
                    else
                        failed=$((failed + 1))
                    fi
                    unset pids[$pid]
                    running=$((running - 1))
                fi
            done
            sleep 1
        done
        
        # Launch new task
        run_task "$idx" "$model" "$press" "$budget" &
        pids[$!]=$!
        running=$((running + 1))
    done
    
    # Wait for all remaining tasks
    for pid in "${pids[@]}"; do
        wait "$pid"
        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    echo ""
    echo "✅ All tasks complete!"
    echo "   Completed: $completed"
    echo "   Failed: $failed"
    echo "   Total: $total_tasks"
}

# Run tasks in parallel
run_parallel_tasks

echo ""
echo "To analyze keyword retention, run:"
echo "  python reason/analyze_keyword_retention.py --result_file <result_file> --model_name <model_name>"

