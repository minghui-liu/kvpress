#!/usr/bin/env python3
"""
Visualize Total Attention Loss Heatmap by Layer and Head.

This creates a heatmap showing how much attention is lost at each layer/head
during KV cache compression.

Usage:
    python visualize_attn_loss_heatmap.py --csv_path path/to/file.csv --output_dir attention_loss_plots
    python visualize_attn_loss_heatmap.py --csv_dir reason/results/csvs --press h2o --output_dir attention_loss_plots
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_and_aggregate_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV and aggregate attention loss by layer and head."""
    df = pd.read_csv(csv_path)
    
    # Calculate attention loss (attn_pre - attn_post)
    df['attn_loss'] = df['attn_pre'] - df['attn_post']
    
    # Aggregate: sum attention loss across all prune steps for each (layer, head)
    agg = df.groupby(['layer_idx', 'head_idx'])['attn_loss'].sum().reset_index()
    
    return agg


def create_heatmap(agg_df: pd.DataFrame, output_path: Path, title: str = "Total Attention Loss Heatmap by Layer and Head"):
    """Create seaborn heatmap of attention loss by layer and head."""
    
    # Pivot to create matrix
    pivot = agg_df.pivot(index='layer_idx', columns='head_idx', values='attn_loss')
    
    # Create figure
    n_layers = pivot.shape[0]
    fig_height = max(8, n_layers * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Create heatmap
    sns.heatmap(
        pivot,
        ax=ax,
        cmap='viridis',
        annot=False,
        fmt='.1f',
        cbar_kws={'label': 'Total Attention Loss'}
    )
    
    ax.set_title(title)
    ax.set_xlabel('head_idx')
    ax.set_ylabel('layer_idx')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return pivot


def main():
    parser = argparse.ArgumentParser(description="Visualize attention loss heatmap by layer and head")
    parser.add_argument("--csv_path", type=str, help="Path to a single CSV file")
    parser.add_argument("--csv_dir", type=str, default="reason/results/csvs", help="Directory containing CSV files")
    parser.add_argument("--press", type=str, help="Filter by press name (e.g., h2o, rkv)")
    parser.add_argument("--model", type=str, help="Filter by model name")
    parser.add_argument("--output_dir", type=str, default="attention_loss_plots", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.csv_path:
        # Process single file
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            return
        
        agg = load_and_aggregate_csv(csv_path)
        output_path = output_dir / f"{csv_path.stem}_attn_loss_heatmap.png"
        
        # Extract press name from filename for title
        press_name = "unknown"
        for p in ["h2o", "rkv_lsh", "rkv", "snapkv", "knorm", "streaming_llm"]:
            if f"__{p}__" in csv_path.name:
                press_name = p
                break
        
        create_heatmap(agg, output_path, title=f"Total Attention Loss Heatmap by Layer and Head ({press_name})")
    
    else:
        # Process all matching files in directory
        csv_dir = Path(args.csv_dir)
        if not csv_dir.exists():
            print(f"Directory not found: {csv_dir}")
            return
        
        csv_files = list(csv_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_path in csv_files:
            # Apply filters
            if args.press and f"__{args.press}__" not in csv_path.name:
                continue
            if args.model and args.model.replace("/", "--") not in csv_path.name:
                continue
            
            print(f"\nProcessing: {csv_path.name}")
            
            try:
                agg = load_and_aggregate_csv(csv_path)
                
                # Extract press name for title
                press_name = "unknown"
                for p in ["h2o", "rkv_lsh", "rkv", "snapkv", "knorm", "streaming_llm"]:
                    if f"__{p}__" in csv_path.name:
                        press_name = p
                        break
                
                output_path = output_dir / f"{csv_path.stem}_attn_loss_heatmap.png"
                create_heatmap(agg, output_path, title=f"Total Attention Loss Heatmap by Layer and Head ({press_name})")
                
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    main()

