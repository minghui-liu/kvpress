#!/usr/bin/env python3
"""
Generate publication-quality heatmap images from token importance scores.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from transformers import AutoTokenizer


def load_json_stream(filepath: Path) -> List[Dict]:
    """Load JSON objects from a file."""
    content = filepath.read_text(encoding='utf-8')
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass
    
    parts = re.split(r'\}\s*\n\s*\{', content.strip())
    results = []
    for i, part in enumerate(parts):
        if i == 0:
            part = part if part.startswith('{') else '{' + part
        else:
            part = '{' + part
        if i == len(parts) - 1:
            part = part if part.endswith('}') else part + '}'
        else:
            part = part + '}'
        try:
            obj = json.loads(part)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            pass
    return results


def normalize_scores_percentile(scores: List[float]) -> List[float]:
    """Normalize scores using percentile ranking."""
    if not scores:
        return []
    non_zero = [s for s in scores if s > 0]
    if not non_zero:
        return [0.0] * len(scores)
    sorted_scores = sorted(set(non_zero))
    rank_map = {s: i / (len(sorted_scores) - 1) if len(sorted_scores) > 1 else 0.5 
                for i, s in enumerate(sorted_scores)}
    return [0.0 if s == 0 else rank_map.get(s, 0.5) for s in scores]


def create_heatmap_image(all_steps: List[Dict], all_token_ids: List[int], 
                         tokenizer, input_len: int, output_path: Path,
                         title: str = "", max_tokens_per_row: int = 50):
    """Create a publication-quality heatmap image."""
    
    # Build score map
    seq_pos_to_score = {}
    for step_data in all_steps:
        cache_to_seq = step_data.get('cache_to_seq_positions', [])
        scores = step_data.get('importance_scores', [])
        if not cache_to_seq or not scores:
            continue
        for cache_pos, seq_pos in enumerate(cache_to_seq):
            if cache_pos < len(scores):
                seq_pos_to_score[seq_pos] = scores[cache_pos]
    
    total_len = len(all_token_ids)
    all_scores = [seq_pos_to_score.get(i, 0.0) for i in range(total_len)]
    normalized = normalize_scores_percentile(all_scores)
    
    # Decode tokens
    tokens_text = [tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) 
                   for tid in all_token_ids]
    
    # Create colormap: Red -> Yellow -> Green
    colors = [(0.8, 0.1, 0.1), (1.0, 1.0, 0.0), (0.1, 0.8, 0.1)]
    cmap = LinearSegmentedColormap.from_list('importance', colors, N=256)
    
    # Calculate layout
    num_rows = (total_len + max_tokens_per_row - 1) // max_tokens_per_row
    
    # Create figure
    fig_height = max(4, num_rows * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Plot tokens
    y_pos = num_rows - 1
    x_pos = 0
    
    for i, (token, norm_score, raw_score) in enumerate(zip(tokens_text, normalized, all_scores)):
        if x_pos >= max_tokens_per_row:
            x_pos = 0
            y_pos -= 1
        
        # Get color
        color = cmap(norm_score)
        
        # Display token (clean up whitespace)
        display = token.replace('\n', '↵').replace('\t', '→')
        if display == ' ':
            display = '·'
        if len(display) > 8:
            display = display[:7] + '…'
        
        # Add rectangle background
        rect = mpatches.FancyBboxPatch(
            (x_pos - 0.45, y_pos - 0.35), 0.9, 0.7,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='#333333' if i >= input_len else '#666666',
            linewidth=0.5 if i < input_len else 1.0
        )
        ax.add_patch(rect)
        
        # Add text
        text_color = 'black' if norm_score > 0.3 else 'white'
        ax.text(x_pos, y_pos, display, ha='center', va='center', 
                fontsize=6, fontfamily='monospace', color=text_color)
        
        x_pos += 1
    
    # Styling
    ax.set_xlim(-0.5, max_tokens_per_row - 0.5)
    ax.set_ylim(-0.5, num_rows - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.02)
    cbar.set_label('Importance Score (Percentile)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Add stats
    evicted = sum(1 for s in all_scores if s == 0)
    stats = f"Total: {total_len} tokens | Input: {input_len} | Generated: {total_len - input_len} | Evicted/Unscored: {evicted}"
    fig.text(0.5, 0.02, stats, ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate heatmap images from token tracking data")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--press", type=str, default="rkv")
    parser.add_argument("--models", "--model", type=str, nargs="+", 
                        default=["deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"])
    parser.add_argument("--question_index", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="heatmap_images")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "pdf"])
    parser.add_argument("--tokens_per_row", type=int, default=60)
    
    args = parser.parse_args()
    
    target_models = [m.replace("/", "--") for m in args.models]
    step_tracking_files = list(Path(args.results_dir).glob("*.step_tracking.json"))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    tokenizer_cache = {}
    
    for step_file in step_tracking_files:
        for target_model in target_models:
            if target_model not in step_file.name:
                continue
            if f"__{args.press}__" not in step_file.name:
                continue
            if f"budget{args.budget}" not in step_file.name:
                continue
            
            print(f"Processing: {step_file.name}")
            questions = load_json_stream(step_file)
            
            question_indices = [args.question_index] if args.question_index is not None else list(range(len(questions)))
            
            model_name = questions[0].get('model_name', '') if questions else ''
            if model_name and model_name not in tokenizer_cache:
                tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                print(f"  Loaded tokenizer for {model_name}")
            tokenizer = tokenizer_cache.get(model_name)
            
            # Load jsonl for full sequences
            jsonl_file = step_file.with_suffix('').with_suffix('.jsonl')
            jsonl_results = []
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        jsonl_results.append(json.loads(line))
            
            for q_idx in question_indices:
                if q_idx >= len(questions):
                    continue
                
                question_data = questions[q_idx]
                generation_steps = question_data.get('generation_steps', [])
                steps_with_scores = [s for s in generation_steps if 'importance_scores' in s and 'cache_to_seq_positions' in s]
                
                if not steps_with_scores:
                    continue
                
                input_text = question_data.get('input_text', '')
                input_token_count = len(tokenizer.encode(input_text, add_special_tokens=True)) if tokenizer else 0
                
                # Get full token sequence
                all_token_ids = []
                if q_idx < len(jsonl_results):
                    response = jsonl_results[q_idx].get('response', '')
                    if tokenizer and response:
                        all_token_ids = tokenizer.encode(input_text + response, add_special_tokens=True)
                
                if not all_token_ids:
                    max_seq = max(max(s.get('cache_to_seq_positions', [0])) for s in steps_with_scores)
                    all_token_ids = list(range(max_seq + 1))
                
                # Generate image
                actual_press = question_data.get('press_name', args.press)
                actual_budget = question_data.get('cache_budget', args.budget)
                model_safe = target_model.replace("/", "_").replace("--", "_")
                
                output_path = output_dir / f"{actual_press}_budget{actual_budget}_q{q_idx}_{model_safe}.{args.format}"
                
                title = f"{actual_press.upper()} (budget={actual_budget}) - Question {q_idx}"
                
                create_heatmap_image(
                    steps_with_scores, all_token_ids, tokenizer, input_token_count,
                    output_path, title=title, max_tokens_per_row=args.tokens_per_row
                )
                
                print(f"  📊 Q{q_idx}: {len(all_token_ids)} tokens → {output_path.name}")
    
    print(f"\n✅ Images saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()


