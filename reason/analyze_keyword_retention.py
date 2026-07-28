"""
Analyze keyword retention from evaluation results.
Reads JSONL files and analyzes how often keywords (names, quantities, objects) are retained.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_keyword_retention_from_results(result_file: Path, model_name: str = ""):
    """
    Analyze keyword retention from evaluation results.
    
    Parameters
    ----------
    result_file : Path
        Path to JSONL result file
    model_name : str
        Retained for CLI compatibility; stored retention metrics do not require a tokenizer.
    """
    # Load results
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Aggregate statistics
    stats = defaultdict(lambda: {
        'total_problems': 0,
        'names_retained': 0,
        'quantities_retained': 0,
        'objects_retained': 0,
        'names_total': 0,
        'quantities_total': 0,
        'objects_total': 0,
    })
    
    skipped_untracked = 0
    overall_retained = 0
    overall_total = 0

    for result in results:
        tracking_status = result.get("retention_tracking_status")
        if tracking_status not in ("tracked", "no_compression"):
            skipped_untracked += 1
            continue
        if 'keyword_retention' not in result:
            continue
        
        retention = result['keyword_retention']
        for key_type in ['names', 'quantities', 'objects']:
            if key_type in retention:
                stats[key_type]['total_problems'] += 1
                stats[key_type][f'{key_type}_retained'] += retention[key_type].get('retained_count', 0)
                stats[key_type][f'{key_type}_total'] += retention[key_type].get('total_count', 0)
                overall_retained += retention[key_type].get('retained_count', 0)
                overall_total += retention[key_type].get('total_count', 0)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Keyword Retention Analysis: {result_file.name}")
    print(f"{'='*60}\n")
    
    for key_type in ['names', 'quantities', 'objects']:
        s = stats[key_type]
        if s['total_problems'] > 0:
            retention_rate = s[f'{key_type}_retained'] / s[f'{key_type}_total'] if s[f'{key_type}_total'] > 0 else 0.0
            print(f"{key_type.capitalize()}:")
            print(f"  Total problems: {s['total_problems']}")
            print(f"  Total keyword tokens: {s[f'{key_type}_total']}")
            print(f"  Retained keyword tokens: {s[f'{key_type}_retained']}")
            print(f"  Retention rate: {retention_rate:.2%}")
            print()

    if overall_total > 0:
        print("All critical keyword categories:")
        print(f"  Retained keyword tokens: {overall_retained} / {overall_total}")
        print(f"  Critical token retention rate: {overall_retained / overall_total:.2%}")
        print()

    if skipped_untracked:
        print(
            f"Warning: skipped {skipped_untracked} result(s) without reliable "
            "retention tracking. Re-run them with --track_tokens=true using a supported press."
        )
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True, help="Path to JSONL result file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Deprecated compatibility argument; no longer required",
    )
    args = parser.parse_args()
    
    analyze_keyword_retention_from_results(Path(args.result_file), args.model_name)
