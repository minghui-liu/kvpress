#!/usr/bin/env python3
"""
Batch experiment runner for TurboQuant (INT4 quantization).
Runs experiments sequentially on local machines (no SLURM).
PRESS_NAME=turboquant, N_BITS=4 (INT4), no token pruning.

Usage:
    python batch_script_turboquant.py                    # Run all experiments
    python batch_script_turboquant.py --range 0-10       # Run experiments 0-10
    python batch_script_turboquant.py --range 5-15       # Run experiments 5-15
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple

# =====================
# Configuration
# =====================
SCRIPT_PATH = "reason/evaluate.py"
RESULT_DIR = "reason/results"

PRESS_NAME = "turboquant"
N_BITS = 4          # INT4 quantization
NUM_SAMPLES = 100
RANDOM_SEED = 42

MODELS = [
   # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
   # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

DATASETS = [
    # "aime24",
    "math500",
    # "gsm8k",
]


def resolve_max_tokens(dataset: str) -> int:
    """Determine max new tokens per dataset (matches batch_script_rkv.py)."""
    if dataset == "math500":
        return 16384
    elif dataset == "aime24":
        return 32768
    else:
        return 2048


def get_experiment_list() -> List[Tuple[str, str]]:
    """Generate list of (model, dataset) tuples."""
    return [(model, dataset) for model in MODELS for dataset in DATASETS]


def run_experiment(model_name: str, dataset: str) -> bool:
    """
    Run a single TurboQuant experiment.
    Returns True if successful, False otherwise.
    """
    model_file = model_name.replace("/", "--")
    max_new_tokens = resolve_max_tokens(dataset)

    score_file = (
        f"{RESULT_DIR}/{dataset}____"
        f"{model_file}__{PRESS_NAME}__"
        f"int{N_BITS}__"
        f"max_new_tokens{max_new_tokens}__"
        f"num_samples{NUM_SAMPLES}__sampling_score.json"
    )

    if os.path.isfile(score_file):
        print(f"✓ Skipping (already completed): {dataset} | {model_name}")
        return True

    print(f"\n{'='*70}")
    print(f"Running: {dataset} | {model_name}")
    print(f"Press: {PRESS_NAME} | Bits: INT{N_BITS} | Max tokens: {max_new_tokens}")
    print(f"{'='*70}")

    cmd = [
        "python",
        SCRIPT_PATH,
        f"--result_dir={RESULT_DIR}",
        f"--dataset={dataset}",
        f"--model_name={model_name}",
        f"--press_name={PRESS_NAME}",
        f"--n_bits={N_BITS}",
        f"--num_samples={NUM_SAMPLES}",
        f"--random_seed={RANDOM_SEED}",
        f"--max_new_tokens={max_new_tokens}",
        f"--track_tokens=false",
        f"--measure_memory=true",
        f"--measure_latency=false",
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed: {dataset} | {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {dataset} | {model_name}")
        print(f"  Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⊘ Interrupted: {dataset} | {model_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run batch TurboQuant (INT4) experiments sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_script_turboquant.py                      # Run all experiments
  python batch_script_turboquant.py --range 0-3          # Run experiments 0-3
        """,
    )
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Range of experiments to run (e.g., '0-3')",
    )
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    all_experiments = get_experiment_list()
    total = len(all_experiments)
    print(f"Total experiments: {total}")
    print(f"Models: {len(MODELS)}, Datasets: {len(DATASETS)}")
    print(f"Settings: PRESS_NAME={PRESS_NAME}, N_BITS=INT{N_BITS}, NUM_SAMPLES={NUM_SAMPLES}")

    if args.range:
        try:
            start, end = map(int, args.range.split("-"))
            if start < 0 or end >= total or start > end:
                print(f"Error: Invalid range {start}-{end}. Valid range is 0-{total-1}")
                sys.exit(1)
            experiments = all_experiments[start: end + 1]
            print(f"\nRunning experiments {start}-{end} ({len(experiments)} total)")
        except (ValueError, IndexError):
            print("Error: Invalid range format. Use 'start-end' (e.g., '0-3')")
            sys.exit(1)
    else:
        experiments = all_experiments
        print(f"\nRunning all {len(experiments)} experiments")

    successful = 0
    failed = 0

    try:
        for i, (model, dataset) in enumerate(experiments, 1):
            experiment_num = all_experiments.index((model, dataset))
            print(f"\n[{i}/{len(experiments)}] Experiment #{experiment_num}")
            if run_experiment(model, dataset):
                successful += 1
            else:
                failed += 1
    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user.")

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
