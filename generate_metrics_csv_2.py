#!/usr/bin/env python3
import json
import os
import csv
from pathlib import Path

# Define the results directory
results_dir = Path("/Users/dixiyao/Desktop/Research/Tahseen/kvpress/results_3")

# Define the mapping for models
model_map = {
    "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": "Llama-8b",
    "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": "Qwen-14B"
}

# Define datasets
datasets = ["math500", "aime24"]
methods = ["full"]  # Only full method in results_3
budgets = [128, 256, 512, 1024]

# Function to read JSON file and extract metrics
def get_metrics(dataset, model, method, budget):
    # Determine max_new_tokens based on dataset
    max_tokens = "16384" if dataset == "math500" else "32768"
    
    # Try different filename patterns for full method in results_3
    patterns = [
        f"{dataset}____{model}__{method}__budget{budget}__max_new_tokens{max_tokens}__num_samples10__sampling_score.json",
        f"{dataset}____{model}__full__budget{budget}__max_new_tokens{max_tokens}__num_samples10__sampling_score.json",
    ]
    
    for pattern in patterns:
        filepath = results_dir / pattern
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    return {
                        'memory': data.get(
                            'avg_decoding_memory_usage_gb',
                            data.get('avg_memory_usage_gb', None),
                        ),
                        'throughput': data.get('avg_output_tokens_per_second', None),
                        'decoding_time': data.get('avg_decoding_time', None),
                        'total_decoding_tokens': data.get('total_decoding_tokens', None)
                    }
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                return None
    
    return None

# Create CSV content
csv_content = []

# Memory (MB) section
csv_content.append(["Memory (MB)", "full", "", "", ""])
for dataset in datasets:
    # Dataset and budget row
    dataset_name = dataset.upper().replace("MATH500", "MATH-500").replace("AIME24", "AIME-24")
    csv_content.append([dataset_name, "128", "256", "512", "1024"])
    
    # Model rows
    for model_key, model_name in model_map.items():
        row = [model_name]
        for method in methods:
            for budget in budgets:
                metrics = get_metrics(dataset, model_key, method, budget)
                if metrics and metrics['memory'] is not None:
                    # Convert GB to MB
                    memory_mb = metrics['memory'] * 1024
                    row.append(f"{memory_mb:.2f}")
                else:
                    row.append("")
        csv_content.append(row)
    
    # Empty row between datasets (except after last dataset)
    csv_content.append([])

# Throughput(tok/s) section
csv_content.append(["Throughput(tok/s)", "", "", "", ""])
for dataset in datasets:
    # Method header row
    csv_content.append(["", "full", "", "", ""])
    # Dataset and budget row
    dataset_name = dataset.upper().replace("MATH500", "MATH-500").replace("AIME24", "AIME-24")
    csv_content.append([dataset_name, "128", "256", "512", "1024"])
    
    # Model rows
    for model_key, model_name in model_map.items():
        row = [model_name]
        for method in methods:
            for budget in budgets:
                metrics = get_metrics(dataset, model_key, method, budget)
                if metrics and metrics['throughput'] is not None:
                    row.append(f"{metrics['throughput']:.2f}")
                else:
                    row.append("")
        csv_content.append(row)
    
    # Empty row between datasets (except after last dataset)
    csv_content.append([])

# Decoding time (s) per token section
csv_content.append(["Decoding time (s) per token", "", "", "", ""])
for dataset in datasets:
    # Method header row
    csv_content.append(["", "full", "", "", ""])
    # Dataset and budget row
    dataset_name = dataset.upper().replace("MATH500", "MATH-500").replace("AIME24", "AIME-24")
    csv_content.append([dataset_name, "128", "256", "512", "1024"])
    
    # Model rows
    for model_key, model_name in model_map.items():
        row = [model_name]
        for method in methods:
            for budget in budgets:
                metrics = get_metrics(dataset, model_key, method, budget)
                if metrics and metrics['decoding_time'] is not None:
                    row.append(f"{metrics['decoding_time']:.2f}")
                else:
                    row.append("")
        csv_content.append(row)
    
    # Empty row between datasets (except after last dataset)
    csv_content.append([])

# Total decoding tokens section
csv_content.append(["Total decoding tokens", "", "", "", ""])
for dataset in datasets:
    # Method header row
    csv_content.append(["", "full", "", "", ""])
    # Dataset and budget row
    dataset_name = dataset.upper().replace("MATH500", "MATH-500").replace("AIME24", "AIME-24")
    csv_content.append([dataset_name, "128", "256", "512", "1024"])
    
    # Model rows
    for model_key, model_name in model_map.items():
        row = [model_name]
        for method in methods:
            for budget in budgets:
                metrics = get_metrics(dataset, model_key, method, budget)
                if metrics and metrics['total_decoding_tokens'] is not None:
                    row.append(f"{metrics['total_decoding_tokens']}")
                else:
                    row.append("")
        csv_content.append(row)
    
    # Empty row between datasets (except after last dataset)
    csv_content.append([])

# Write to CSV file
output_file = results_dir.parent / "metrics_output_2.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_content)

print(f"CSV file created: {output_file}")
print(f"\nYou can now copy the contents to Google Sheets!")

# Also print a preview
print("\nPreview of first few rows:")
for i, row in enumerate(csv_content[:10]):
    print(','.join(str(x) for x in row))
