#!/usr/bin/env python3
"""
Batch experiment runner for local machines (no SLURM).
Runs experiments sequentially and allows selecting a range.
Supports both max token modes: "separate" and "force2048"

For every (press, model, dataset, budget, n_hash_buckets) combo, this also sweeps top_p in
TOP_P_LIST and repeats each top_p NUM_RUNS times (run1..run4) using the SAME fixed RANDOM_SEED
(so the dataset sample selection is identical across repeats) with do_sample=True and temperature
fixed at TEMPERATURE (0.6) - the point of the repeats is to measure sampling variance, since
generation itself is not seeded.

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

PRESS_NAMES = ["rkv", "h2o", "knorm", "snapkv", "streaming_llm","scope","rpc"]

MODELS = [
    # "meta-llama/Llama-3.1-8B-Instruct",  # ML
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  # DQ
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",  # LN
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # DL
]

DATASETS = [
    # "aime24",
    "math500",
    # "gsm8k",
    # "drop",
    # "reclor",
    # "folio"
]

CACHE_BUDGETS = [128,256,384,512] #1024
LAMBDA = 0.1  # Match batch.sh (was 0.01)
N_HASH_BUCKETS_LIST = [6]
N_BITS = 4  # evaluate.py default, only used for turboquant filenames
SNAPKV_WINDOW_SIZE = 64  # evaluate.py default, only used for snapkv/pyramidkv filenames
RANDOM_SEED = 42  # Fixed across all runs/top_p values - only generation sampling varies

TEMPERATURE = 0.6
TOP_P_LIST = [0.9, 0.95, 1.0]
NUM_RUNS = 4  # run1..run4, same seed/config, to measure sampling variance

# Max tokens modes to traverse (match batch.sh)
MAX_TOKENS_MODES = ["separate"]

# Dataset-specific NUM_SAMPLES
NUM_SAMPLES_MAP = {
    "aime24": 0,  # Match batch.sh
    "math500": 100,
    "gsm8k": 100,
    "drop":100,
    "reclor": 100,
    "folio": 100,
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
            return 32768
    else:
        return 32768


def format_lambda(lam: float) -> str:
    """
    Format lambda for filenames. Mirrors reason/evaluate.py's sanitizer exactly (bug-for-bug,
    e.g. lam=0.1 -> "010", not "01") so skip_existing/score_file lookups actually match the
    filenames evaluate.py writes.
    """
    lam_int = int(round(lam * 100))
    if lam_int == 0:
        lam_sanitized = "0"
    elif lam_int < 10:
        lam_sanitized = f"00{lam_int}"
    elif lam_int < 100:
        lam_sanitized = f"0{lam_int}"
    else:
        lam_sanitized = str(lam_int)
        lam_sanitized = str(int(lam_sanitized) // 100) if lam_int % 100 == 0 else lam_sanitized
    return lam_sanitized


def format_top_p(top_p: float) -> str:
    """Format top_p for filenames. Mirrors reason/evaluate.py's __topp{...} suffix exactly."""
    return f"{top_p:.3f}".rstrip("0").rstrip(".").replace(".", "")


def get_experiment_list() -> List[Tuple[str, str, str, int, int, float, int]]:
    """
    Generate list of (press_name, model, dataset, budget, n_hash_buckets, top_p, run_id) tuples.
    run_id is 1-indexed (run1..run4). Returns all possible experiment combinations.
    """
    experiments = []
    for press_name in PRESS_NAMES:
        for model in MODELS:
            for dataset in DATASETS:
                for budget in CACHE_BUDGETS:
                    for n_buckets in N_HASH_BUCKETS_LIST:
                        for top_p in TOP_P_LIST:
                            for run_id in range(1, NUM_RUNS + 1):
                                experiments.append((press_name, model, dataset, budget, n_buckets, top_p, run_id))
    return experiments


def build_filenames(
    dataset: str,
    model_name: str,
    press_name: str,
    cache_budget: int,
    n_hash_buckets: int,
    max_new_tokens: int,
    num_samples: int,
    top_p: float,
    run_tag: str,
) -> Tuple[str, str]:
    """
    Build (out_file, score_file) paths exactly matching what reason/evaluate.py writes, so
    skip_existing checks below actually line up. Must be kept in sync with evaluate.py's
    save_filename construction.
    """
    model_file = model_name.replace("/", "--")
    lambda_sanitized = format_lambda(LAMBDA)
    top_p_sanitized = format_top_p(top_p)

    # Base name: mirrors the if/elif/else press_name branches in evaluate.py
    if "rkv" in press_name:
        stem = (
            f"{dataset}____{model_file}__{press_name}__"
            f"budget{cache_budget}__hash_bucket{n_hash_buckets}__"
            f"max_new_tokens{max_new_tokens}__lam{lambda_sanitized}"
        )
    elif press_name == "turboquant":
        stem = f"{dataset}____{model_file}__{press_name}__int{N_BITS}__max_new_tokens{max_new_tokens}"
    elif press_name in ("snapkv", "snapkv_press", "pyramidkv"):
        stem = (
            f"{dataset}____{model_file}__{press_name}__"
            f"budget{cache_budget}__window{SNAPKV_WINDOW_SIZE}__max_new_tokens{max_new_tokens}"
        )
    else:
        # Covers h2o, knorm, streaming_llm, scope, rpc, random, full, none
        stem = f"{dataset}____{model_file}__{press_name}__budget{cache_budget}__max_new_tokens{max_new_tokens}"

    # Suffixes: mirrors the trailing appends in evaluate.py (num_samples/fraction, seed, sampling, topp, run_tag)
    if num_samples > 0:
        stem += f"__num_samples{num_samples}"
    if num_samples > 0:
        stem += f"__seed{RANDOM_SEED}"
    stem += "__sampling"  # do_sampling is always True here
    stem += f"__topp{top_p_sanitized}"
    if run_tag:
        stem += f"__{run_tag}"

    out_file = f"{RESULT_DIR}/{stem}.jsonl"
    score_file = f"{RESULT_DIR}/{stem}_score.json"
    return out_file, score_file


def run_experiment(
    model_name: str,
    dataset: str,
    cache_budget: int,
    press_name: str,
    max_tokens_mode: str,
    n_hash_buckets: int,
    top_p: float,
    run_id: int,
) -> tuple[bool, float]:
    """
    Run a single experiment.
    Returns (success: bool, elapsed_time: float in seconds).
    """
    max_new_tokens = resolve_max_tokens(dataset, max_tokens_mode)
    num_samples = NUM_SAMPLES_MAP.get(dataset, 100)
    run_tag = f"run{run_id}"

    out_file, score_file = build_filenames(
        dataset, model_name, press_name, cache_budget, n_hash_buckets, max_new_tokens, num_samples, top_p, run_tag
    )

    # Check if already completed
    if os.path.isfile(score_file):
        print(f"✓ Skipping (already completed): {dataset} @ budget {cache_budget} | top_p {top_p} | {run_tag}")
        return True, 0.0

    print(f"\n{'='*70}")
    print(f"Running: {dataset} | {model_name} | budget {cache_budget} | buckets {n_hash_buckets}")
    print(f"Max tokens: {max_new_tokens} | Max tokens mode: {max_tokens_mode} | Press: {press_name}")
    print(f"Temperature: {TEMPERATURE} | top_p: {top_p} | {run_tag} | seed: {RANDOM_SEED} (fixed)")
    print(f"Output: {out_file}")
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
        f"--n_hash_buckets={n_hash_buckets}",
        f"--lam={LAMBDA}",
        f"--temperature={TEMPERATURE}",
        f"--top_p={top_p}",
        f"--run_tag={run_tag}",
        f"--track_tokens=false",
        f"--enable_qualitative_analysis=false",
        f"--measure_memory=false",
        f"--measure_latency=false",
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"✓ Completed: {dataset} @ budget {cache_budget} | top_p {top_p} | {run_tag} (Time: {elapsed_time:.2f}s)")
        return True, elapsed_time
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed: {dataset} @ budget {cache_budget} | top_p {top_p} | {run_tag} (Time: {elapsed_time:.2f}s)")
        print(f"  Error: {e}")
        return False, elapsed_time
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\n⊘ Interrupted: {dataset} @ budget {cache_budget} | top_p {top_p} | {run_tag} (Time: {elapsed_time:.2f}s)")
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
    print(f"Press methods: {len(PRESS_NAMES)}, Models: {len(MODELS)}, Datasets: {len(DATASETS)}, Budgets: {len(CACHE_BUDGETS)}, Hash buckets: {N_HASH_BUCKETS_LIST}")
    print(f"Top-p values: {TOP_P_LIST}, Runs per config: {NUM_RUNS} (fixed seed {RANDOM_SEED}), Temperature: {TEMPERATURE}")

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
            press_name, model, dataset, budget, n_buckets, top_p, run_id = all_experiments[combo]

            print(
                f"\n[{idx}/{len(range_list)}] Task #{task_id} "
                f"(Mode: {max_tokens_mode}, Press: {press_name}, Buckets: {n_buckets}, top_p: {top_p}, run: {run_id})"
            )

            success, elapsed_time = run_experiment(
                model, dataset, budget, press_name, max_tokens_mode, n_buckets, top_p, run_id
            )
            total_wall_time += elapsed_time
            experiment_times.append((press_name, model, dataset, budget, n_buckets, top_p, run_id, elapsed_time))
            
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
        print(f"  Min: {min(t[7] for t in experiment_times):.2f}s")
        print(f"  Max: {max(t[7] for t in experiment_times):.2f}s")
        print(f"\nTop 5 slowest experiments:")
        sorted_times = sorted(experiment_times, key=lambda x: x[7], reverse=True)
        for i, (press, model, dataset, budget, n_buckets, top_p, run_id, elapsed) in enumerate(sorted_times[:5], 1):
            model_short = model.split("/")[-1][:20]
            print(
                f"  {i}. {press:15} | {model_short:20} | {dataset:10} | budget{budget:4d} | "
                f"buckets{n_buckets:3d} | top_p{top_p:.2f} | run{run_id} | {elapsed:7.2f}s"
            )
    print(f"{'='*70}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
