#!/usr/bin/env python3
"""
Batch experiment runner for RKV LSH method (matches batch_slurm.sh settings).
Runs experiments sequentially on local machines (no SLURM).
PRESS_NAME=rkvlsh, LAMBDA=0.1, NUM_SAMPLES=0, MAX_TOKENS_MODE=separate

Usage:
    python batch_script_rkv.py                    # Run all experiments
    python batch_script_rkv.py --range 0-10       # Run experiments 0-10
    python batch_script_rkv.py --range 5-15       # Run experiments 5-15
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple

# =====================
# Configuration (matches batch_slurm.sh)
# =====================
SCRIPT_PATH = "reason/evaluate.py"
RESULT_DIR = "reason/results"

PRESS_NAME = "rkvlsh"
LAMBDA = 0.1  # Different from batch.sh
N_HASH_BUCKETS = 8
NUM_SAMPLES = 0
RANDOM_SEED = 42

MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]

DATASETS = [
    "aime24",
    "math500",
]

CACHE_BUDGETS = [128, 256, 512, 1024]

# Max tokens mode - batch_slurm.sh only uses "separate" mode
MAX_TOKENS_MODES = ["separate"]


def resolve_max_tokens(dataset: str, mode: str) -> int:
    """Determine max tokens based on mode and dataset."""
    if mode == "force2048":
        return 2048
    elif mode == "separate":
        if dataset == "math500":
            return 16384
        elif dataset == "aime24":
            return 32768
        else:
            return 2048
    else:
        return 2048


def format_lambda(lam: float) -> str:
    """Format lambda for filenames."""
    lambda_int = int(round(lam * 100))
    if lambda_int == 0:
        return "0"
    elif lambda_int < 10:
        return f"{lambda_int:03d}"
    elif lambda_int < 100:
        return f"{lambda_int:02d}"
    else:
        if lambda_int % 100 == 0:
            return str(lambda_int // 100)
        else:
            return str(lambda_int)


def get_experiment_list() -> List[Tuple[str, str, int]]:
    """
    Generate list of (model, dataset, budget) tuples.
    Returns all possible experiment combinations.
    """
    experiments = []
    for model in MODELS:
        for dataset in DATASETS:
            for budget in CACHE_BUDGETS:
                experiments.append((model, dataset, budget))
    return experiments


def run_experiment(
    model_name: str,
    dataset: str,
    cache_budget: int,
    press_name: str,
    max_tokens_mode: str,
) -> bool:
    """
    Run a single experiment.
    Returns True if successful, False otherwise.
    """
    model_file = model_name.replace("/", "--")
    max_new_tokens = resolve_max_tokens(dataset, max_tokens_mode)
    num_samples = NUM_SAMPLES
    lambda_sanitized = format_lambda(LAMBDA)

    # Generate output filenames
    out_file = (
        f"{RESULT_DIR}/{dataset}____"
        f"{model_file}__{press_name}__"
        f"budget{cache_budget}__"
        f"hash_bucket{N_HASH_BUCKETS}__"
        f"max_new_tokens{max_new_tokens}__"
        f"lam{lambda_sanitized}__"
        f"num_samples{num_samples}__sampling.jsonl"
    )
    
    score_file = (
        f"{RESULT_DIR}/{dataset}____"
        f"{model_file}__{press_name}__"
        f"budget{cache_budget}__"
        f"hash_bucket{N_HASH_BUCKETS}__"
        f"max_new_tokens{max_new_tokens}__"
        f"lam{lambda_sanitized}__"
        f"num_samples{num_samples}__sampling_score.json"
    )

    # Check if already completed
    if os.path.isfile(score_file):
        print(f"✓ Skipping (already completed): {dataset} @ budget {cache_budget}")
        return True

    print(f"\n{'='*70}")
    print(f"Running: {dataset} | {model_name} | budget {cache_budget}")
    print(f"Max tokens: {max_new_tokens} | Max tokens mode: {max_tokens_mode} | Press: {press_name} | Lambda: {LAMBDA}")
    print(f"{'='*70}")

    # Build command
    cmd = [
        "python",
        SCRIPT_PATH,
        f"--dataset={dataset}",
        f"--model_name={model_name}",
        f"--press_name={press_name}",
        f"--cache_budget={cache_budget}",
        f"--num_samples={num_samples}",
        f"--random_seed={RANDOM_SEED}",
        f"--max_new_tokens={max_new_tokens}",
        f"--n_hash_buckets={N_HASH_BUCKETS}",
        f"--lam={LAMBDA}",
        f"--track_tokens=false",
        f"--measure_memory=false",
        f"--measure_latency=true",
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Completed: {dataset} @ budget {cache_budget}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {dataset} @ budget {cache_budget}")
        print(f"  Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⊘ Interrupted: {dataset} @ budget {cache_budget}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run batch RKV LSH experiments sequentially on local machine (matches batch_slurm.sh)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_script_rkv.py                      # Run all experiments
  python batch_script_rkv.py --range 0-10         # Run experiments 0-10
  python batch_script_rkv.py --range 5-15         # Run experiments 5-15
        """,
    )
    
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Range of experiments to run (e.g., '0-10' or '5-20')",
    )

    args = parser.parse_args()

    # Setup environment
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Get all experiments
    all_experiments = get_experiment_list()
    total = len(all_experiments)
    print(f"Total possible experiments: {total}")
    print(f"Models: {len(MODELS)}, Datasets: {len(DATASETS)}, Budgets: {len(CACHE_BUDGETS)}")
    print(f"Settings: PRESS_NAME={PRESS_NAME}, LAMBDA={LAMBDA}, NUM_SAMPLES={NUM_SAMPLES}")

    # Determine range
    if args.range:
        try:
            start, end = args.range.split("-")
            start = int(start)
            end = int(end)
            if start < 0 or end >= total or start > end:
                print(f"Error: Invalid range {start}-{end}. Valid range is 0-{total-1}")
                sys.exit(1)
            experiments = all_experiments[start : end + 1]
            print(f"\nRunning experiments {start}-{end} ({len(experiments)} total)")
        except (ValueError, IndexError):
            print(f"Error: Invalid range format. Use 'start-end' (e.g., '0-10')")
            sys.exit(1)
    else:
        experiments = all_experiments
        print(f"\nRunning all {len(experiments)} experiments")

    # Run experiments
    successful = 0
    failed = 0

    try:
        for i, (model, dataset, budget) in enumerate(experiments, 1):
            experiment_num = all_experiments.index((model, dataset, budget))
            print(f"\n[{i}/{len(experiments)}] Experiment #{experiment_num}")
            
            result = run_experiment(model, dataset, budget, PRESS_NAME, "separate")
            if result:
                successful += 1
            else:
                failed += 1

    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user.")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total run: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*70}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
