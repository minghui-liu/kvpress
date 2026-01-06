#!/usr/bin/env python3
"""
Analyze LSH bucket distribution for RKV-LSH method.
Runs 5 samples with Qwen-14B on math500, budget 128, and tracks bucket counts.
Generates histogram visualization.

Usage:
    python analyze_bucket_distribution.py
"""

import argparse
import subprocess
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# =====================
# Configuration
# =====================
SCRIPT_PATH = "reason/evaluate.py"
RESULT_DIR = "bucket_analysis_results"
PRESS_NAME = "rkvlsh"

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET = "math500"
CACHE_BUDGET = 128
LAMBDA = 0
N_HASH_BUCKETS = 8
NUM_BUCKETS = 2 ** N_HASH_BUCKETS  # 256 buckets
RANDOM_SEED = 42
NUM_SAMPLES = 1
MAX_NEW_TOKENS = 16384  # math500 default


def run_experiment_with_tracking() -> str:
    """
    Run experiment with bucket tracking enabled.
    Returns path to the output file.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    model_file = MODEL.replace("/", "--")
    lambda_sanitized = "0"
    
    out_file = (
        f"{RESULT_DIR}/{DATASET}____"
        f"{model_file}__{PRESS_NAME}__"
        f"budget{CACHE_BUDGET}__"
        f"hash_bucket{N_HASH_BUCKETS}__"
        f"max_new_tokens{MAX_NEW_TOKENS}__"
        f"lam{lambda_sanitized}__"
        f"num_samples{NUM_SAMPLES}__bucket_analysis.jsonl"
    )
    
    score_file = out_file.replace(".jsonl", "_score.json")
    
    # Check if already completed
    if os.path.isfile(score_file) and os.path.isfile(out_file):
        print(f"✓ Results already exist: {out_file}")
        return out_file
    
    print(f"\n{'='*70}")
    print(f"Running bucket analysis experiment")
    print(f"Dataset: {DATASET} | Model: {MODEL}")
    print(f"Budget: {CACHE_BUDGET} | Buckets: {NUM_BUCKETS}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"{'='*70}\n")
    
    # Build command with bucket tracking flag
    cmd = [
        "python",
        SCRIPT_PATH,
        "--model_name", MODEL,
        "--dataset", DATASET,
        "--press_method", PRESS_NAME,
        "--cache_budget", str(CACHE_BUDGET),
        "--n_hash_buckets", str(N_HASH_BUCKETS),
        "--lam", str(LAMBDA),
        "--num_samples", str(NUM_SAMPLES),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--seed", str(RANDOM_SEED),
        "--out_file", out_file,
        "--score_file", score_file,
        "--track_buckets",  # Enable bucket tracking
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Experiment completed successfully")
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed with exit code {e.returncode}")
        sys.exit(1)


def load_bucket_data(jsonl_file: str) -> dict:
    """
    Load bucket counts from JSONL output file.
    Returns dict mapping sample_id -> bucket_counts array.
    """
    bucket_data = {}
    
    if not os.path.exists(jsonl_file):
        print(f"Error: File not found: {jsonl_file}")
        return bucket_data
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'bucket_counts' in data:
                    sample_id = data.get('sample_id', len(bucket_data))
                    bucket_counts = np.array(data['bucket_counts'])
                    bucket_data[sample_id] = bucket_counts
    
    return bucket_data


def plot_bucket_histograms(bucket_data: dict, output_dir: str = "bucket_visualizations"):
    """
    Generate histogram plots for bucket distributions.
    Creates both per-sample plots and aggregated plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not bucket_data:
        print("No bucket data to plot")
        return
    
    num_samples = len(bucket_data)
    print(f"\nGenerating histograms for {num_samples} samples...")
    
    # Aggregate all bucket counts
    all_counts = np.zeros(NUM_BUCKETS)
    for counts in bucket_data.values():
        all_counts += counts
    
    # Plot 1: Individual sample histograms in grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (sample_id, counts) in enumerate(sorted(bucket_data.items())):
        if idx >= 5:
            break
        ax = axes[idx]
        ax.bar(range(NUM_BUCKETS), counts, width=1.0, color='steelblue', alpha=0.7)
        ax.set_title(f"Sample {sample_id}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Bucket ID", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.tick_params(axis='both', labelsize=9)
    
    # Plot aggregated in last subplot
    ax = axes[5]
    ax.bar(range(NUM_BUCKETS), all_counts, width=1.0, color='coral', alpha=0.7)
    ax.set_title("Aggregated (All Samples)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Bucket ID", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(axis='both', labelsize=9)
    
    fig.suptitle(f"LSH Bucket Distribution - {DATASET} - {MODEL.split('/')[-1]} - Budget {CACHE_BUDGET}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = f"{output_dir}/bucket_histogram_grid.pdf"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved grid histogram: {output_path}")
    plt.close()
    
    # Plot 2: Single large aggregated histogram
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.bar(range(NUM_BUCKETS), all_counts, width=1.0, color='steelblue', alpha=0.7)
    ax.set_title(f"Aggregated LSH Bucket Distribution - {DATASET} - Budget {CACHE_BUDGET} - {NUM_SAMPLES} samples", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Bucket ID", fontsize=13)
    ax.set_ylabel("Total Count", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    
    # Add statistics text
    stats_text = f"Total tokens: {int(all_counts.sum())}\n"
    stats_text += f"Mean per bucket: {all_counts.mean():.1f}\n"
    stats_text += f"Std dev: {all_counts.std():.1f}\n"
    stats_text += f"Max: {int(all_counts.max())}\n"
    stats_text += f"Min: {int(all_counts.min())}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    output_path = f"{output_dir}/bucket_histogram_aggregated.pdf"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved aggregated histogram: {output_path}")
    plt.close()
    
    # Plot 3: Distribution statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution of counts
    ax = axes[0]
    ax.hist(all_counts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_title("Distribution of Bucket Counts", fontsize=12, fontweight='bold')
    ax.set_xlabel("Count Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)
    
    # Top N buckets
    ax = axes[1]
    top_n = 20
    top_indices = np.argsort(all_counts)[-top_n:][::-1]
    top_counts = all_counts[top_indices]
    ax.barh(range(top_n), top_counts, color='coral', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"Bucket {i}" for i in top_indices], fontsize=9)
    ax.set_title(f"Top {top_n} Buckets by Count", fontsize=12, fontweight='bold')
    ax.set_xlabel("Count", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.3, axis='x')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    output_path = f"{output_dir}/bucket_statistics.pdf"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved statistics plot: {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("BUCKET DISTRIBUTION STATISTICS")
    print("="*70)
    print(f"Total buckets: {NUM_BUCKETS}")
    print(f"Total tokens: {int(all_counts.sum())}")
    print(f"Mean tokens per bucket: {all_counts.mean():.2f}")
    print(f"Std deviation: {all_counts.std():.2f}")
    print(f"Min bucket count: {int(all_counts.min())}")
    print(f"Max bucket count: {int(all_counts.max())}")
    print(f"Empty buckets: {int((all_counts == 0).sum())}")
    print(f"Buckets with >100 tokens: {int((all_counts > 100).sum())}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze LSH bucket distribution")
    parser.add_argument("--skip-run", action="store_true", 
                       help="Skip running experiment, only visualize existing data")
    parser.add_argument("--output-dir", type=str, default="bucket_visualizations",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    # Run experiment (or skip if requested)
    if args.skip_run:
        # Construct expected filename
        model_file = MODEL.replace("/", "--")
        out_file = (
            f"{RESULT_DIR}/{DATASET}____"
            f"{model_file}__{PRESS_NAME}__"
            f"budget{CACHE_BUDGET}__"
            f"hash_bucket{N_HASH_BUCKETS}__"
            f"max_new_tokens{MAX_NEW_TOKENS}__"
            f"lam0__"
            f"num_samples{NUM_SAMPLES}__bucket_analysis.jsonl"
        )
        print(f"Skipping experiment run, using existing file: {out_file}")
    else:
        out_file = run_experiment_with_tracking()
    
    # Load bucket data
    print(f"\nLoading bucket data from: {out_file}")
    bucket_data = load_bucket_data(out_file)
    
    if not bucket_data:
        print("Error: No bucket data found in output file")
        print("Make sure the experiment was run with --track_buckets flag")
        sys.exit(1)
    
    print(f"Loaded bucket data for {len(bucket_data)} samples")
    
    # Generate visualizations
    plot_bucket_histograms(bucket_data, output_dir=args.output_dir)
    
    print(f"\n✓ Analysis complete! Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
