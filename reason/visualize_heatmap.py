#!/usr/bin/env python3
"""
Visualize importance scores as a heatmap showing which tokens are retained vs evicted.

Usage:
    python visualize_heatmap.py --results_dir reason/results --model meta-llama/Meta-Llama-3.1-8B-Instruct --budget 128 --press rkv
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Will generate text-based visualization only.")


def load_json_stream(filepath: Path) -> List[Dict]:
    """Load JSON objects from a file (one per entry or as stream)."""
    content = filepath.read_text(encoding='utf-8')
    # Try loading as single JSON first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        # Try loading as JSON lines
        results = []
        for line in content.strip().split('\n'):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return results


def create_text_heatmap(step_data: Dict, max_tokens_per_line: int = 20) -> str:
    """Create a text-based heatmap visualization."""
    lines = []
    
    step_num = step_data.get('step', 0)
    all_tokens = step_data.get('all_tokens_text', [])
    scores = step_data.get('importance_scores', [])
    retained_positions = set(step_data.get('retained_positions', []))
    
    if not scores or not all_tokens:
        return f"Step {step_num}: No score data available\n"
    
    lines.append(f"\n{'='*80}")
    lines.append(f"COMPRESSION STEP {step_num}")
    lines.append(f"Total tokens: {len(all_tokens)}, Retained: {len(retained_positions)}, Evicted: {len(all_tokens) - len(retained_positions)}")
    lines.append(f"{'='*80}")
    
    # Normalize scores for visualization
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score != min_score else 1
    
    # Create token display with scores
    lines.append("\nToken importance (▓=high, ░=low, [EVICTED] for removed tokens):\n")
    
    current_line = []
    for pos, (token, score) in enumerate(zip(all_tokens, scores)):
        normalized = (score - min_score) / score_range
        
        # Create visual indicator
        if normalized > 0.8:
            indicator = "▓▓▓"
        elif normalized > 0.6:
            indicator = "▓▓░"
        elif normalized > 0.4:
            indicator = "▓░░"
        elif normalized > 0.2:
            indicator = "░░░"
        else:
            indicator = "···"
        
        # Mark if evicted
        if pos not in retained_positions:
            token_display = f"[{token[:8]}]✗"
        else:
            token_display = f" {token[:8]} "
        
        entry = f"{indicator}{token_display}"
        current_line.append(entry)
        
        if len(current_line) >= max_tokens_per_line:
            lines.append("  ".join(current_line))
            current_line = []
    
    if current_line:
        lines.append("  ".join(current_line))
    
    # Show top retained and top evicted
    lines.append(f"\n{'─'*40}")
    
    # Sort by score
    token_scores = list(zip(range(len(all_tokens)), all_tokens, scores))
    token_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Top retained (high scores that were kept)
    top_retained = [(pos, tok, sc) for pos, tok, sc in token_scores if pos in retained_positions][:10]
    lines.append("\n🟢 TOP RETAINED (highest importance):")
    for pos, tok, sc in top_retained:
        lines.append(f"   pos={pos:3d} score={sc:.4f}  '{tok}'")
    
    # Top evicted (lowest scores that were removed)
    evicted = [(pos, tok, sc) for pos, tok, sc in token_scores if pos not in retained_positions]
    evicted.sort(key=lambda x: x[2])  # Sort ascending (lowest first)
    top_evicted = evicted[:10]
    lines.append("\n🔴 TOP EVICTED (lowest importance):")
    for pos, tok, sc in top_evicted:
        lines.append(f"   pos={pos:3d} score={sc:.4f}  '{tok}'")
    
    return "\n".join(lines)


def create_matplotlib_heatmap(question_data: Dict, output_path: Path):
    """Create a matplotlib heatmap visualization."""
    if not HAS_MATPLOTLIB:
        return
    
    generation_steps = question_data.get('generation_steps', [])
    if not generation_steps:
        print("No generation steps found for matplotlib visualization")
        return
    
    # Find the step with scores
    steps_with_scores = [s for s in generation_steps if 'importance_scores' in s]
    if not steps_with_scores:
        print("No steps with importance scores found")
        return
    
    # Create figure with subplots for each compression step
    n_steps = len(steps_with_scores)
    fig, axes = plt.subplots(n_steps, 1, figsize=(16, 4 * n_steps), squeeze=False)
    
    for idx, step_data in enumerate(steps_with_scores):
        ax = axes[idx, 0]
        
        scores = step_data.get('importance_scores', [])
        all_tokens = step_data.get('all_tokens_text', [])
        retained_positions = set(step_data.get('retained_positions', []))
        
        if not scores:
            continue
        
        # Create heatmap data
        scores_array = np.array(scores).reshape(1, -1)
        
        # Create custom colormap: red (low/evicted) -> yellow -> green (high/retained)
        cmap = mcolors.LinearSegmentedColormap.from_list("retention", ["#ff4444", "#ffff44", "#44ff44"])
        
        # Plot heatmap
        im = ax.imshow(scores_array, aspect='auto', cmap=cmap)
        
        # Mark evicted positions with X
        for pos in range(len(scores)):
            if pos not in retained_positions:
                ax.axvline(x=pos, color='red', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f"Compression Step {step_data.get('step', idx)}: {len(retained_positions)} retained, {len(scores) - len(retained_positions)} evicted")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Importance")
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Importance Score")
        
        # Add token labels for a subset (too many would be unreadable)
        if len(all_tokens) <= 50:
            ax.set_xticks(range(len(all_tokens)))
            ax.set_xticklabels([t[:6] for t in all_tokens], rotation=90, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize importance scores as heatmap")
    parser.add_argument("--budget", type=int, default=128, help="Cache budget to filter")
    parser.add_argument("--press", type=str, default="rkv", help="Press name to filter")
    parser.add_argument("--models", "--model", type=str, nargs="+", 
                        default=["meta-llama--Meta-Llama-3.1-8B-Instruct"],
                        help="Model names to process")
    parser.add_argument("--question_index", type=int, default=0, help="Question index to visualize")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--output_dir", type=str, default="heatmap_visualizations", help="Output directory")
    
    args = parser.parse_args()
    
    target_budget = args.budget
    target_press = args.press
    target_models = [m.replace("/", "--") for m in args.models]
    question_index = args.question_index
    
    # Find step tracking files
    step_tracking_files = list(Path(args.results_dir).glob("*.step_tracking.json"))
    
    if not step_tracking_files:
        print(f"No step_tracking.json files found in {args.results_dir}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for step_file in step_tracking_files:
        for target_model in target_models:
            if target_model not in step_file.name:
                continue
            if target_press not in step_file.name:
                continue
            if f"budget{target_budget}" not in step_file.name:
                continue
            
            print(f"\nProcessing: {step_file.name}")
            
            questions = load_json_stream(step_file)
            
            if question_index >= len(questions):
                print(f"  Question index {question_index} not found (only {len(questions)} questions)")
                continue
            
            question_data = questions[question_index]
            generation_steps = question_data.get('generation_steps', [])
            
            if not generation_steps:
                print("  No generation steps found")
                continue
            
            # Check if any step has scores
            steps_with_scores = [s for s in generation_steps if 'importance_scores' in s]
            if not steps_with_scores:
                print("  No importance scores found in tracking data.")
                print("  Re-run evaluation with --track_tokens=True to capture scores.")
                continue
            
            print(f"  Found {len(steps_with_scores)} steps with importance scores")
            
            # Generate text visualization
            txt_lines = []
            txt_lines.append(f"Question: {question_data.get('input_text', '')[:100]}...")
            txt_lines.append(f"Model: {question_data.get('model_name', 'unknown')}")
            txt_lines.append(f"Press: {question_data.get('press_name', 'unknown')}")
            txt_lines.append(f"Cache Budget: {question_data.get('cache_budget', 'unknown')}")
            
            for step_data in steps_with_scores:
                txt_lines.append(create_text_heatmap(step_data))
            
            # Save text visualization
            model_safe = target_model.replace("/", "_").replace("--", "_")
            txt_path = output_dir / f"heatmap_q{question_index}_{target_press}_budget{target_budget}_{model_safe}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(txt_lines))
            print(f"  📝 Saved text heatmap to: {txt_path}")
            
            # Generate matplotlib visualization if available
            if HAS_MATPLOTLIB:
                png_path = output_dir / f"heatmap_q{question_index}_{target_press}_budget{target_budget}_{model_safe}.png"
                create_matplotlib_heatmap(question_data, png_path)
    
    print("\n✅ Heatmap visualization complete!")


if __name__ == "__main__":
    main()

