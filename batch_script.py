#!/usr/bin/env python3
"""
Batch experiment runner for local machines (no SLURM).
Runs experiments sequentially and allows selecting a range.
Supports both max token modes: "separate" and "force2048"

Usage:
    python batch_script.py                                # Run all experiments (all modes)
    python batch_script.py --range 0-10                   # Run experiments 0-10
    python batch_script.py --range 5-20                   # Run experiments 5-20
    python batch_script.py --mode separate                # Run only separate mode
    python batch_script.py --mode force2048               # Run only force2048 mode
    python batch_script.py --range 0-7 --mode separate
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple

# =====================
# Configuration
# =====================
SCRIPT_PATH = "reason/evaluate.py"
RESULT_DIR = "reason/results"

PRESS_NAMES = ["rkvlsh"]

MODELS = [
    # "meta-llama/Llama-3.1-8B-Instruct",  # ML
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  # DQ
    #"nvidia/Llama-3.1-Nemotron-Nano-8B-v1",  # LN
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # DL
]

DATASETS = [
    # "aime24",
    "math500",
    #"gsm8k",
]

CACHE_BUDGETS = [1024] #1024
LAMBDA = 0.1  # Match batch.sh (was 0.01)
N_HASH_BUCKETS = 8
RANDOM_SEED = 42

# Max tokens modes to traverse (match batch.sh)
MAX_TOKENS_MODES = ["separate"]

# Dataset-specific NUM_SAMPLES
NUM_SAMPLES_MAP = {
    "aime24": 0,  # Match batch.sh
    "math500": 500,
    "gsm8k": 100,
}


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


def get_experiment_list() -> List[Tuple[str, str, str, int]]:
    """
    Generate list of (press_name, model, dataset, budget) tuples.
    Returns all possible experiment combinations.
    """
    experiments = []
    for press_name in PRESS_NAMES:
        for model in MODELS:
            for dataset in DATASETS:
                for budget in CACHE_BUDGETS:
                    experiments.append((press_name, model, dataset, budget))
    return experiments


def run_experiment(
    model_name: str,
    dataset: str,
    cache_budget: int,
    press_name: str,
    max_tokens_mode: str,
) -> tuple[bool, float]:
    """
    Run a single experiment.
    Returns (success: bool, elapsed_time: float in seconds).
    """
    model_file = model_name.replace("/", "--")
    max_new_tokens = resolve_max_tokens(dataset, max_tokens_mode)
    num_samples = NUM_SAMPLES_MAP.get(dataset, 100)
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
    print(f"Max tokens: {max_new_tokens} | Max tokens mode: {max_tokens_mode} | Press: {press_name}")
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
        f"--enable_qualitative_analysis=false",
        f"--measure_memory=false",
        f"--measure_latency=false",
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"✓ Completed: {dataset} @ budget {cache_budget} (Time: {elapsed_time:.2f}s)")
        return True, elapsed_time
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed: {dataset} @ budget {cache_budget} (Time: {elapsed_time:.2f}s)")
        print(f"  Error: {e}")
        return False, elapsed_time
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\n⊘ Interrupted: {dataset} @ budget {cache_budget} (Time: {elapsed_time:.2f}s)")
        return False, elapsed_time


def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments sequentially on local machine with max tokens modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_script.py                              # Run all experiments (all modes)
  python batch_script.py --range 0-10                 # Run experiments 0-10
  python batch_script.py --range 5-15                 # Run experiments 5-15
  python batch_script.py --mode separate              # Run only separate mode
  python batch_script.py --mode force2048             # Run only force2048 mode
  python batch_script.py --range 0-7 --mode separate  # Run 0-7 in separate mode
        """,
    )
    
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Range of experiments to run (e.g., '0-10' or '5-20')",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=MAX_TOKENS_MODES,
        help=f"Max tokens mode to use: {', '.join(MAX_TOKENS_MODES)} (default: all modes)",
    )

    args = parser.parse_args()

    # Setup environment
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Determine which modes to run
    if args.mode:
        modes_to_run = [args.mode]
    else:
        modes_to_run = MAX_TOKENS_MODES
    
    # Get all experiments (per mode)
    all_experiments = get_experiment_list()
    experiments_per_mode = len(all_experiments)
    total_experiments = experiments_per_mode * len(modes_to_run)
    
    print(f"Modes to run: {modes_to_run}")
    print(f"Total possible experiments: {total_experiments} ({len(modes_to_run)} modes × {experiments_per_mode} experiments)")
    print(f"Press methods: {len(PRESS_NAMES)}, Models: {len(MODELS)}, Datasets: {len(DATASETS)}, Budgets: {len(CACHE_BUDGETS)}")

    # Determine range
    if args.range:
        try:
            start, end = args.range.split("-")
            start = int(start)
            end = int(end)
            if start < 0 or end >= total_experiments or start > end:
                print(f"Error: Invalid range {start}-{end}. Valid range is 0-{total_experiments-1}")
                sys.exit(1)
            range_list = list(range(start, end + 1))
            print(f"\nRunning experiments {start}-{end} ({len(range_list)} total)")
        except (ValueError, IndexError):
            print(f"Error: Invalid range format. Use 'start-end' (e.g., '0-10')")
            sys.exit(1)
    else:
        range_list = list(range(total_experiments))
        print(f"\nRunning all {total_experiments} experiments")

    # Run experiments
    successful = 0
    failed = 0
    total_wall_time = 0.0
    experiment_times = []

    try:
        for idx, task_id in enumerate(range_list, 1):
            # Map task_id to (mode, press_name, model, dataset, budget)
            mode_idx = task_id // experiments_per_mode
            combo = task_id % experiments_per_mode
            
            # Check if this mode is in modes_to_run
            if mode_idx >= len(modes_to_run):
                continue
            
            max_tokens_mode = modes_to_run[mode_idx]
            press_name, model, dataset, budget = all_experiments[combo]
            
            print(f"\n[{idx}/{len(range_list)}] Task #{task_id} (Mode: {max_tokens_mode}, Press: {press_name})")
            
            success, elapsed_time = run_experiment(model, dataset, budget, press_name, max_tokens_mode)
            total_wall_time += elapsed_time
            experiment_times.append((press_name, model, dataset, budget, elapsed_time))
            
            if success:
                successful += 1
            else:
                failed += 1

    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user.")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments run: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if experiment_times:
        print(f"\nWall Clock Times:")
        print(f"  Total: {total_wall_time:.2f}s ({total_wall_time/60:.2f} min)")
        print(f"  Average per experiment: {total_wall_time/(successful+failed):.2f}s")
        print(f"  Min: {min(t[4] for t in experiment_times):.2f}s")
        print(f"  Max: {max(t[4] for t in experiment_times):.2f}s")
        print(f"\nTop 5 slowest experiments:")
        sorted_times = sorted(experiment_times, key=lambda x: x[4], reverse=True)
        for i, (press, model, dataset, budget, elapsed) in enumerate(sorted_times[:5], 1):
            model_short = model.split("/")[-1][:20]
            print(f"  {i}. {press:15} | {model_short:20} | {dataset:10} | budget{budget:4d} | {elapsed:7.2f}s")
    print(f"{'='*70}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
