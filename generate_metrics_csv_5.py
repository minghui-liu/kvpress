#!/usr/bin/env python3
"""Generate metrics CSV from results_5 folder."""

import json
import os
from pathlib import Path
import csv


def detect_methods(results_dir):
    """Detect which methods have results in the directory."""
    methods = set()
    for file in os.listdir(results_dir):
        if file.endswith('_score.json'):
            # Format: dataset____model__method__...
            # Split by __ but need to handle the ____ between dataset and model
            parts = file.split('__')
            # Skip empty parts from ____
            parts = [p for p in parts if p]
            # parts[0] = dataset, parts[1] = model, parts[2] = method
            if len(parts) >= 3:
                method = parts[2]
                methods.add(method)
    return sorted(methods)


def get_metrics(results_dir, dataset, model, method, budget, max_tokens):
    """Extract metrics from score file."""
    # Construct filename pattern based on method
    if method == 'full':
        pattern = f"{dataset}____{model}__full__max_new_tokens{max_tokens}__num_samples"
    else:
        # For rkvlsh with hash_bucket8 and lam0
        pattern = f"{dataset}____{model}__{method}__budget{budget}__hash_bucket8__max_new_tokens{max_tokens}__lam0__num_samples"
    
    # Find matching files
    for file in os.listdir(results_dir):
        if file.startswith(pattern) and file.endswith('_score.json'):
            file_path = os.path.join(results_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract metrics with fallback
                    # Prefer decoding-only peak memory. Fall back to the legacy
                    # combined generation peak for older result files.
                    memory_gb = data.get(
                        'avg_decoding_memory_usage_gb',
                        data.get('avg_memory_usage_gb', 0),
                    )
                    memory_mb = memory_gb * 1024  # Convert GB to MB
                    throughput = data.get('avg_output_tokens_per_second', 0)
                    decoding_time = data.get('avg_decoding_time', 0)
                    total_tokens = data.get('total_decoding_tokens', 0)
                    
                    return {
                        'memory_mb': round(memory_mb, 2),
                        'throughput': round(throughput, 2),
                        'decoding_time': round(decoding_time, 2),
                        'total_tokens': total_tokens
                    }
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
    return None


def main():
    results_dir = 'results_5'
    
    # Detect methods
    methods = detect_methods(results_dir)
    print(f"Detected methods: {methods}")
    
    # Configuration
    datasets = ['math500', 'aime24']
    models = [
        'deepseek-ai--DeepSeek-R1-Distill-Llama-8B',
        'deepseek-ai--DeepSeek-R1-Distill-Qwen-14B'
    ]
    budgets = [128, 256, 512, 1024]
    
    # Define max_tokens based on dataset
    dataset_max_tokens = {
        'math500': [2048, 16384],  # separate mode uses 16384 for math500
        'aime24': [2048, 32768]    # separate mode uses 32768 for aime24
    }
    
    # Create CSV output
    output_file = 'metrics_output_5.csv'
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['Dataset', 'Model', 'Method', 'Budget', 'Max_Tokens', 
                 'Memory_MB', 'Throughput', 'Decoding_Time', 'Total_Tokens']
        writer.writerow(header)
        
        # Iterate through all combinations
        for dataset in datasets:
            max_tokens_list = dataset_max_tokens[dataset]
            for model in models:
                for method in methods:
                    if method == 'full':
                        # Full method doesn't have budget
                        for max_tokens in max_tokens_list:
                            metrics = get_metrics(results_dir, dataset, model, method, None, max_tokens)
                            if metrics:
                                row = [
                                    dataset,
                                    model,
                                    method,
                                    'N/A',
                                    max_tokens,
                                    metrics['memory_mb'],
                                    metrics['throughput'],
                                    metrics['decoding_time'],
                                    metrics['total_tokens']
                                ]
                                writer.writerow(row)
                    else:
                        # Methods with budget
                        for budget in budgets:
                            for max_tokens in max_tokens_list:
                                metrics = get_metrics(results_dir, dataset, model, method, budget, max_tokens)
                                if metrics:
                                    row = [
                                        dataset,
                                        model,
                                        method,
                                        budget,
                                        max_tokens,
                                        metrics['memory_mb'],
                                        metrics['throughput'],
                                        metrics['decoding_time'],
                                        metrics['total_tokens']
                                    ]
                                    writer.writerow(row)
    
    print(f"CSV file generated: {output_file}")
    print(f"Total rows written: {sum(1 for _ in open(output_file)) - 1}")  # -1 for header


if __name__ == '__main__':
    main()
