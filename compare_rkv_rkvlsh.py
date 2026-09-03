#!/usr/bin/env python3
"""
Compare qualitative token retention decisions between RKV and RKV-LSH.

Identifies interesting cases where:
1. RKV-LSH drops meaningful tokens that RKV retains
2. RKV-LSH drops repetitive tokens that RKV fails to drop

Usage:
    python compare_rkv_rkvlsh.py \
        --rkv_file ranking_analysis/qualitative_analysis_lam1.0_buckets6.json \
        --rkvlsh_file ranking_analysis/qualitative_analysis_lam0.0_buckets32.json \
        --output report.txt
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np


def load_qualitative_data(json_file):
    """Load qualitative analysis data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def is_repetitive_token(token_text, all_tokens, position):
    """
    Heuristic to detect if a token is repetitive.
    Checks if the same token appears frequently nearby.
    """
    # Count occurrences in a window around this position
    window_size = 20
    start = max(0, position - window_size)
    end = min(len(all_tokens), position + window_size)

    window_tokens = [all_tokens[i]['text'] for i in range(start, end)]
    token_count = window_tokens.count(token_text)

    # If token appears more than 3 times in the window, consider it repetitive
    return token_count > 3


def is_meaningful_token(token_text):
    """
    Heuristic to detect if a token is potentially meaningful.
    Looks for content words (not just punctuation/whitespace).
    """
    # Strip whitespace
    stripped = token_text.strip()

    # Empty or pure whitespace
    if not stripped:
        return False

    # Pure punctuation
    if all(c in '.,;:!?-–—()[]{}"\'' for c in stripped):
        return False

    # Very common function words (less meaningful)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'be'}
    if stripped.lower() in common_words:
        return False

    # Has alphanumeric content
    return any(c.isalnum() for c in stripped)


def compare_samples(rkv_sample, rkvlsh_sample):
    """
    Compare token retention decisions between RKV and RKV-LSH for a single sample.

    Returns:
        dict with analysis results including:
        - meaningful_dropped_by_lsh: tokens RKV kept but RKV-LSH dropped (potentially meaningful)
        - repetitive_dropped_by_lsh: repetitive tokens RKV-LSH dropped but RKV kept
        - agreement_rate: percentage of tokens where both methods agreed
    """
    # Get retained positions for each method
    rkv_retained = {t['position'] for t in rkv_sample['tokens'] if t['retained']}
    rkvlsh_retained = {t['position'] for t in rkvlsh_sample['tokens'] if t['retained']}

    # Find differences
    only_rkv = rkv_retained - rkvlsh_retained  # RKV kept, RKV-LSH dropped
    only_rkvlsh = rkvlsh_retained - rkv_retained  # RKV-LSH kept, RKV dropped

    # Analyze tokens RKV kept but RKV-LSH dropped
    meaningful_dropped = []
    repetitive_dropped = []

    for pos in only_rkv:
        token = next(t for t in rkv_sample['tokens'] if t['position'] == pos)
        token_text = token['text']

        # Check if meaningful
        if is_meaningful_token(token_text):
            # Get scores from both methods
            rkvlsh_token = next(t for t in rkvlsh_sample['tokens'] if t['position'] == pos)
            meaningful_dropped.append({
                'position': pos,
                'text': token_text,
                'rkv_score': token['final_score'],
                'rkvlsh_score': rkvlsh_token['final_score'],
                'rkvlsh_attention': rkvlsh_token.get('attention_score'),
                'rkvlsh_redundancy': rkvlsh_token.get('redundancy_score'),
            })

        # Check if repetitive
        if is_repetitive_token(token_text, rkv_sample['tokens'], pos):
            rkvlsh_token = next(t for t in rkvlsh_sample['tokens'] if t['position'] == pos)
            repetitive_dropped.append({
                'position': pos,
                'text': token_text,
                'rkv_score': token['final_score'],
                'rkvlsh_score': rkvlsh_token['final_score'],
                'rkvlsh_attention': rkvlsh_token.get('attention_score'),
                'rkvlsh_redundancy': rkvlsh_token.get('redundancy_score'),
            })

    # Calculate agreement rate
    total_positions = len(rkv_sample['tokens'])
    agreements = total_positions - len(only_rkv) - len(only_rkvlsh)
    agreement_rate = agreements / total_positions if total_positions > 0 else 0

    return {
        'sample_id': rkv_sample['sample_id'],
        'meaningful_dropped_by_lsh': meaningful_dropped,
        'repetitive_dropped_by_lsh': repetitive_dropped,
        'only_rkv_count': len(only_rkv),
        'only_rkvlsh_count': len(only_rkvlsh),
        'agreement_rate': agreement_rate,
        'total_tokens': total_positions,
    }


def generate_report(comparisons, output_file):
    """Generate a human-readable report of the comparison."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RKV vs RKV-LSH Qualitative Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        total_samples = len(comparisons)
        avg_agreement = np.mean([c['agreement_rate'] for c in comparisons])
        total_meaningful = sum(len(c['meaningful_dropped_by_lsh']) for c in comparisons)
        total_repetitive = sum(len(c['repetitive_dropped_by_lsh']) for c in comparisons)

        f.write(f"Total Samples Analyzed: {total_samples}\n")
        f.write(f"Average Agreement Rate: {avg_agreement:.2%}\n")
        f.write(f"Total Meaningful Tokens Dropped by RKV-LSH: {total_meaningful}\n")
        f.write(f"Total Repetitive Tokens Dropped by RKV-LSH: {total_repetitive}\n\n")

        # Find most interesting samples
        f.write("=" * 80 + "\n")
        f.write("CASE A: Examples where RKV-LSH drops meaningful tokens\n")
        f.write("=" * 80 + "\n\n")

        # Sort by number of meaningful tokens dropped
        meaningful_samples = sorted(comparisons,
                                   key=lambda x: len(x['meaningful_dropped_by_lsh']),
                                   reverse=True)

        for i, sample in enumerate(meaningful_samples[:5], 1):  # Top 5
            if not sample['meaningful_dropped_by_lsh']:
                continue

            f.write(f"\n--- Example {i}: Sample {sample['sample_id']} ---\n")
            f.write(f"Agreement Rate: {sample['agreement_rate']:.2%}\n")
            f.write(f"Meaningful Tokens Dropped: {len(sample['meaningful_dropped_by_lsh'])}\n\n")

            # Show top 10 meaningful tokens
            for token in sample['meaningful_dropped_by_lsh'][:10]:
                f.write(f"  Position {token['position']}: '{token['text']}'\n")
                f.write(f"    RKV Score (kept): {token['rkv_score']:.4f}\n")
                f.write(f"    RKV-LSH Score (dropped): {token['rkvlsh_score']:.4f}\n")
                if token['rkvlsh_attention'] is not None:
                    f.write(f"    RKV-LSH Attention: {token['rkvlsh_attention']:.4f}, "
                           f"Redundancy: {token['rkvlsh_redundancy']:.4f}\n")
                f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CASE B: Examples where RKV-LSH drops repetitive tokens\n")
        f.write("=" * 80 + "\n\n")

        # Sort by number of repetitive tokens dropped
        repetitive_samples = sorted(comparisons,
                                   key=lambda x: len(x['repetitive_dropped_by_lsh']),
                                   reverse=True)

        for i, sample in enumerate(repetitive_samples[:5], 1):  # Top 5
            if not sample['repetitive_dropped_by_lsh']:
                continue

            f.write(f"\n--- Example {i}: Sample {sample['sample_id']} ---\n")
            f.write(f"Agreement Rate: {sample['agreement_rate']:.2%}\n")
            f.write(f"Repetitive Tokens Dropped: {len(sample['repetitive_dropped_by_lsh'])}\n\n")

            # Show top 10 repetitive tokens
            for token in sample['repetitive_dropped_by_lsh'][:10]:
                f.write(f"  Position {token['position']}: '{token['text']}'\n")
                f.write(f"    RKV Score (kept): {token['rkv_score']:.4f}\n")
                f.write(f"    RKV-LSH Score (dropped): {token['rkvlsh_score']:.4f}\n")
                if token['rkvlsh_attention'] is not None:
                    f.write(f"    RKV-LSH Attention: {token['rkvlsh_attention']:.4f}, "
                           f"Redundancy: {token['rkvlsh_redundancy']:.4f}\n")
                f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Sample Summary\n")
        f.write("=" * 80 + "\n\n")

        for sample in comparisons:
            f.write(f"Sample {sample['sample_id']}: "
                   f"Agreement={sample['agreement_rate']:.2%}, "
                   f"Meaningful Dropped={len(sample['meaningful_dropped_by_lsh'])}, "
                   f"Repetitive Dropped={len(sample['repetitive_dropped_by_lsh'])}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare RKV and RKV-LSH token retention decisions")
    parser.add_argument("--rkv_file", type=str, required=True,
                       help="Path to RKV qualitative analysis JSON file")
    parser.add_argument("--rkvlsh_file", type=str, required=True,
                       help="Path to RKV-LSH qualitative analysis JSON file")
    parser.add_argument("--output", type=str, default="rkv_comparison_report.txt",
                       help="Output report file path")

    args = parser.parse_args()

    # Load data
    print(f"Loading RKV data from {args.rkv_file}...")
    rkv_data = load_qualitative_data(args.rkv_file)

    print(f"Loading RKV-LSH data from {args.rkvlsh_file}...")
    rkvlsh_data = load_qualitative_data(args.rkvlsh_file)

    # Compare samples
    print("Comparing samples...")
    comparisons = []
    for rkv_sample, rkvlsh_sample in zip(rkv_data, rkvlsh_data):
        assert rkv_sample['sample_id'] == rkvlsh_sample['sample_id'], \
            "Sample IDs don't match! Make sure both files are from the same run."
        comparison = compare_samples(rkv_sample, rkvlsh_sample)
        comparisons.append(comparison)

    # Generate report
    print(f"Generating report...")
    generate_report(comparisons, args.output)

    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
