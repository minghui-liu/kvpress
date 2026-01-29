#!/usr/bin/env python3
"""
Visualize token importance as colored text - each word colored by its attention score.
Red = low importance (evicted), Green = high importance (retained)

Usage:
    python visualize_text_heatmap.py --results_dir reason/results --model meta-llama/Meta-Llama-3.1-8B-Instruct --budget 128 --press rkv
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer


def load_json_stream(filepath: Path) -> List[Dict]:
    """Load JSON objects from a file (handles multiple concatenated JSON objects)."""
    content = filepath.read_text(encoding='utf-8')
    
    # Try loading as single JSON first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass
    
    # Split on '}\n{' pattern (pretty-printed concatenated JSON)
    import re
    parts = re.split(r'\}\s*\n\s*\{', content.strip())
    
    results = []
    for i, part in enumerate(parts):
        # Add back the braces we split on
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


def score_to_color(normalized: float) -> str:
    """Convert normalized score (0-1) to RGB color. Red=low, Yellow=mid, Green=high."""
    normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
    
    # Red (low) -> Yellow (mid) -> Green (high)
    if normalized < 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (normalized * 2))
        b = 0
    else:
        # Yellow to Green
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
        b = 0
    
    return f"rgb({r},{g},{b})"


def normalize_scores_percentile(scores: List[float]) -> List[float]:
    """Normalize scores using percentile ranking (0-1 range).
    
    This ensures colors are spread evenly across the score distribution,
    regardless of the absolute score values.
    """
    if not scores:
        return []
    
    # Handle all-zero case
    non_zero = [s for s in scores if s > 0]
    if not non_zero:
        return [0.0] * len(scores)
    
    # Sort scores and create rank mapping
    sorted_scores = sorted(set(non_zero))
    rank_map = {s: i / (len(sorted_scores) - 1) if len(sorted_scores) > 1 else 0.5 
                for i, s in enumerate(sorted_scores)}
    
    # Map each score to its percentile rank (0 stays at 0)
    normalized = []
    for s in scores:
        if s == 0:
            normalized.append(0.0)  # Evicted tokens stay red
        else:
            normalized.append(rank_map.get(s, 0.5))
    
    return normalized


def score_to_ansi(score: float, min_score: float, max_score: float) -> str:
    """Convert importance score to ANSI color code for terminal."""
    if max_score == min_score:
        normalized = 0.5
    else:
        normalized = (score - min_score) / (max_score - min_score)
    
    # Use 256-color mode: red (196) -> yellow (226) -> green (46)
    if normalized < 0.33:
        return "\033[48;5;196m"  # Red background
    elif normalized < 0.66:
        return "\033[48;5;226m\033[30m"  # Yellow background, black text
    else:
        return "\033[48;5;46m\033[30m"  # Green background, black text


def create_html_heatmap(input_text: str, step_data: Dict, tokenizer, step_num: int, input_token_count: int = None) -> str:
    """Create HTML with colored tokens based on SUM of importance scores across all heads.
    
    Shows ALL tokens (input + generated) with their importance scores.
    Red = low importance, Green = high importance.
    """
    
    # Get ALL tokens and scores
    all_tokens_text = step_data.get('all_tokens_text', [])
    scores = step_data.get('importance_scores', [])
    input_len = step_data.get('input_length', input_token_count or 0)
    
    if not all_tokens_text:
        return f"<p>Step {step_num}: No token data available</p>"
    
    if not scores:
        return f"<p>Step {step_num}: No importance scores available</p>"
    
    display_len = len(all_tokens_text)
    all_scores = scores[:display_len] if len(scores) >= display_len else scores + [0] * (display_len - len(scores))
    
    min_score = min(all_scores) if all_scores else 0
    max_score = max(all_scores) if all_scores else 1
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # Separate stats for input vs generated
    input_scores_list = all_scores[:input_len] if input_len > 0 else []
    gen_scores_list = all_scores[input_len:] if input_len < len(all_scores) else []
    
    input_avg = sum(input_scores_list) / len(input_scores_list) if input_scores_list else 0
    gen_avg = sum(gen_scores_list) / len(gen_scores_list) if gen_scores_list else 0
    
    # Normalize scores for better color distribution
    normalized_scores = normalize_scores_percentile(all_scores)
    
    # Build HTML
    html_parts = []
    html_parts.append(f'<div style="margin: 20px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">')
    html_parts.append(f'<h3>Compression Step {step_num}</h3>')
    html_parts.append(f'<p>Total: {display_len} tokens | '
                      f'<span style="color: #88f;">Input: {input_len} (avg: {input_avg:.4f})</span> | '
                      f'<span style="color: #8f8;">Generated: {display_len - input_len} (avg: {gen_avg:.4f})</span></p>')
    html_parts.append(f'<p>Score range: Min={min_score:.4f} | Max={max_score:.4f}</p>')
    html_parts.append('<div style="font-family: monospace; line-height: 1.8; background: #1a1a1a; padding: 15px; border-radius: 5px;">')
    
    for pos in range(display_len):
        token_text = all_tokens_text[pos] if pos < len(all_tokens_text) else "?"
        score = all_scores[pos] if pos < len(all_scores) else 0
        norm_score = normalized_scores[pos] if pos < len(normalized_scores) else 0
        
        is_generated = pos >= input_len
        
        # Color based on percentile-normalized score (red=low, green=high)
        color = score_to_color(norm_score)
        
        # Add underline for generated tokens
        border = "border-bottom: 2px solid #8f8;" if is_generated else ""
        
        # Escape HTML special characters
        display_text = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if not display_text.strip():
            display_text = display_text.replace(' ', '·')
        
        token_type = "GENERATED" if is_generated else "INPUT"
        title = f"pos={pos} score={score:.6f} [{token_type}]"
        
        html_parts.append(
            f'<span style="background-color: {color}; color: black; padding: 2px 1px; {border}'
            f'border-radius: 2px;" title="{title}">{display_text}</span>'
        )
    
    html_parts.append('</div>')
    
    # Add legend
    html_parts.append('<div style="margin-top: 10px; font-size: 12px;">')
    html_parts.append('<span style="background: rgb(255,0,0); color: white; padding: 2px 5px;">Low score (sum across heads)</span> ')
    html_parts.append('<span style="background: rgb(255,255,0); color: black; padding: 2px 5px;">Medium</span> ')
    html_parts.append('<span style="background: rgb(0,255,0); color: black; padding: 2px 5px;">High score</span> ')
    html_parts.append('<span style="border-bottom: 2px solid #8f8; padding: 2px 5px;">Generated token</span>')
    html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)


def create_full_sequence_heatmap(all_steps: List[Dict], all_token_ids: List[int], tokenizer, input_len: int) -> str:
    """Create HTML showing ALL tokens from the full generation with last known importance scores.
    
    For each token position in the full sequence:
    - Use the last known importance score from when it was in the cache
    - If token was evicted and never scored again, use 0
    """
    
    # Build a map: sequence_position -> (last_score, last_step)
    seq_pos_to_score = {}
    
    for step_idx, step_data in enumerate(all_steps):
        cache_to_seq = step_data.get('cache_to_seq_positions', [])
        scores = step_data.get('importance_scores', [])
        
        if not cache_to_seq or not scores:
            continue
        
        # Map each cache position's score to its sequence position
        for cache_pos, seq_pos in enumerate(cache_to_seq):
            if cache_pos < len(scores):
                seq_pos_to_score[seq_pos] = scores[cache_pos]
    
    # Get total sequence length
    total_len = len(all_token_ids)
    
    # Build scores array for full sequence
    all_scores = []
    for seq_pos in range(total_len):
        score = seq_pos_to_score.get(seq_pos, 0.0)  # 0 if never scored or evicted
        all_scores.append(score)
    
    # Decode all tokens
    all_tokens_text = [tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in all_token_ids]
    
    # Stats
    min_score = min(all_scores) if all_scores else 0
    max_score = max(all_scores) if all_scores else 1
    nonzero_scores = [s for s in all_scores if s > 0]
    avg_score = sum(nonzero_scores) / len(nonzero_scores) if nonzero_scores else 0
    evicted_count = sum(1 for s in all_scores if s == 0)
    
    input_scores = all_scores[:input_len]
    gen_scores = all_scores[input_len:]
    input_evicted = sum(1 for s in input_scores if s == 0)
    gen_evicted = sum(1 for s in gen_scores if s == 0)
    
    # Normalize scores using percentile ranking for better color distribution
    normalized_scores = normalize_scores_percentile(all_scores)
    
    # Build HTML
    html_parts = []
    html_parts.append(f'<div style="margin: 20px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">')
    html_parts.append(f'<h3>Full Generation View ({total_len} tokens)</h3>')
    html_parts.append(f'<p>Input: {input_len} tokens ({input_evicted} evicted/unscored) | '
                      f'Generated: {total_len - input_len} tokens ({gen_evicted} evicted/unscored)</p>')
    html_parts.append(f'<p>Score range: Min={min_score:.4f} | Max={max_score:.4f} | Avg (non-zero)={avg_score:.4f}</p>')
    html_parts.append(f'<p><em>Colors use percentile normalization for better contrast</em></p>')
    html_parts.append('<div style="font-family: monospace; line-height: 1.8; background: #1a1a1a; padding: 15px; border-radius: 5px;">')
    
    for pos in range(total_len):
        token_text = all_tokens_text[pos] if pos < len(all_tokens_text) else "?"
        score = all_scores[pos]
        norm_score = normalized_scores[pos] if pos < len(normalized_scores) else 0
        
        is_generated = pos >= input_len
        is_evicted = score == 0
        
        # Color based on percentile-normalized score (red=low/evicted, green=high)
        color = score_to_color(norm_score)
        
        # Add underline for generated tokens
        border = "border-bottom: 2px solid #8f8;" if is_generated else ""
        
        # Escape HTML special characters
        display_text = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '↵ ')
        # Keep spaces visible
        if display_text == ' ':
            display_text = '·'
        
        token_type = "GENERATED" if is_generated else "INPUT"
        evicted_str = " EVICTED" if is_evicted else ""
        title = f"seq_pos={pos} score={score:.6f} [{token_type}]{evicted_str}"
        
        html_parts.append(
            f'<span style="background-color: {color}; color: black; padding: 2px 1px; {border}'
            f'border-radius: 2px; display: inline;" title="{title}">{display_text}</span>'
        )
    
    html_parts.append('</div>')
    
    # Add legend
    html_parts.append('<div style="margin-top: 10px; font-size: 12px;">')
    html_parts.append('<span style="background: rgb(255,0,0); color: white; padding: 2px 5px;">Low score / Evicted (0)</span> ')
    html_parts.append('<span style="background: rgb(255,255,0); color: black; padding: 2px 5px;">Medium</span> ')
    html_parts.append('<span style="background: rgb(0,255,0); color: black; padding: 2px 5px;">High score</span> ')
    html_parts.append('<span style="border-bottom: 2px solid #8f8; padding: 2px 5px;">Generated token</span>')
    html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)


def create_terminal_heatmap(step_data: Dict, step_num: int, input_token_count: int = None) -> str:
    """Create terminal-colored text based on average importance scores."""
    
    all_tokens_text = step_data.get('all_tokens_text', [])
    original_input_tokens_text = step_data.get('original_input_tokens_text', [])
    scores = step_data.get('importance_scores', [])
    
    # Use original input tokens if available
    if original_input_tokens_text:
        tokens_to_show = original_input_tokens_text
    elif all_tokens_text and input_token_count:
        tokens_to_show = all_tokens_text[:input_token_count]
    else:
        return f"Step {step_num}: No data available\n"
    
    if not scores:
        return f"Step {step_num}: No scores available\n"
    
    display_count = len(tokens_to_show)
    input_scores = scores[:display_count] if len(scores) >= display_count else scores
    min_score = min(input_scores) if input_scores else 0
    max_score = max(input_scores) if input_scores else 1
    avg_score = sum(input_scores) / len(input_scores) if input_scores else 0
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"COMPRESSION STEP {step_num}")
    lines.append(f"Input: {display_count} tokens | Avg: {avg_score:.4f} | Min: {min_score:.4f} | Max: {max_score:.4f}")
    lines.append(f"{'='*80}")
    lines.append("Legend: \033[48;5;196m RED=low \033[0m \033[48;5;226m\033[30m YELLOW=mid \033[0m \033[48;5;46m\033[30m GREEN=high \033[0m")
    lines.append("")
    
    # Build colored text - just show scores, no eviction markers
    colored_text = []
    for pos in range(display_count):
        token_text = tokens_to_show[pos] if pos < len(tokens_to_show) else "?"
        score = input_scores[pos] if pos < len(input_scores) else 0
        
        color_code = score_to_ansi(score, min_score, max_score)
        reset = "\033[0m"
        colored_text.append(f"{color_code}{token_text}{reset}")
    
    full_text = ''.join(colored_text)
    lines.append(full_text)
    lines.append("")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize token importance as colored text")
    parser.add_argument("--budget", type=int, default=128, help="Cache budget to filter")
    parser.add_argument("--press", type=str, default="rkv", help="Press name to filter")
    parser.add_argument("--models", "--model", type=str, nargs="+", 
                        default=["meta-llama--Meta-Llama-3.1-8B-Instruct"],
                        help="Model names to process")
    parser.add_argument("--question_index", type=int, default=None, help="Question index (None = all questions)")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--output_dir", type=str, default="text_heatmaps", help="Output directory")
    parser.add_argument("--terminal", action="store_true", help="Print to terminal with ANSI colors")
    
    args = parser.parse_args()
    
    target_budget = args.budget
    target_press = args.press
    target_models = [m.replace("/", "--") for m in args.models]
    question_index = args.question_index
    
    step_tracking_files = list(Path(args.results_dir).glob("*.step_tracking.json"))
    
    print(f"Looking for files in: {Path(args.results_dir).resolve()}")
    print(f"Found {len(step_tracking_files)} step_tracking.json files")
    for f in step_tracking_files:
        print(f"  - {f.name}")
    
    if not step_tracking_files:
        print(f"No step_tracking.json files found in {args.results_dir}")
        return
    
    print(f"\nFiltering for: model={target_models}, press={target_press}, budget={target_budget}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    tokenizer_cache = {}
    matched_any = False
    
    for step_file in step_tracking_files:
        for target_model in target_models:
            if target_model not in step_file.name:
                continue
            # Use exact match for press (avoid rkv matching rkv_lsh)
            if f"__{target_press}__" not in step_file.name:
                continue
            if f"budget{target_budget}" not in step_file.name:
                continue
            matched_any = True
            
            print(f"\nProcessing: {step_file.name}")
            
            questions = load_json_stream(step_file)
            
            # Determine which questions to process
            if args.question_index is not None:
                question_indices = [args.question_index]
            else:
                question_indices = list(range(len(questions)))
            
            print(f"  Found {len(questions)} questions, processing {len(question_indices)}")
            
            # Load tokenizer once
            model_name = questions[0].get('model_name', '') if questions else ''
            if model_name and model_name not in tokenizer_cache:
                try:
                    tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    print(f"  Loaded tokenizer for {model_name}")
                except Exception as e:
                    print(f"  Error loading tokenizer: {e}")
                    continue
            
            tokenizer = tokenizer_cache.get(model_name)
            
            # Load all jsonl results for this file
            jsonl_file = step_file.with_suffix('').with_suffix('.jsonl')
            jsonl_results = []
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        jsonl_results.append(json.loads(line))
            
            for q_idx in question_indices:
                if q_idx >= len(questions):
                    print(f"  Question index {q_idx} not found (only {len(questions)} questions)")
                    continue
                
                question_data = questions[q_idx]
                generation_steps = question_data.get('generation_steps', [])
                
                steps_with_scores = [s for s in generation_steps if 'importance_scores' in s and 'cache_to_seq_positions' in s]
                if not steps_with_scores:
                    print(f"  Q{q_idx}: No importance scores found")
                    continue
                
                input_text = question_data.get('input_text', '')
                
                # Calculate input token count
                input_token_count = None
                if tokenizer and input_text:
                    input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
                    input_token_count = len(input_tokens)
                
                # Get full token sequence from jsonl
                all_token_ids = []
                if q_idx < len(jsonl_results):
                    result_data = jsonl_results[q_idx]
                    response = result_data.get('response', '')
                    if tokenizer and response:
                        full_text = input_text + response
                        all_token_ids = tokenizer.encode(full_text, add_special_tokens=True)
                
                if not all_token_ids and tokenizer:
                    max_seq_pos = 0
                    for step in steps_with_scores:
                        cache_to_seq = step.get('cache_to_seq_positions', [])
                        if cache_to_seq:
                            max_seq_pos = max(max_seq_pos, max(cache_to_seq))
                    all_token_ids = list(range(max_seq_pos + 1))
                
                # Generate HTML
                html_parts = []
                html_parts.append('<!DOCTYPE html>')
                html_parts.append('<html><head>')
                html_parts.append('<meta charset="UTF-8">')
                html_parts.append(f'<title>Q{q_idx} - {target_press} budget{target_budget}</title>')
                html_parts.append('<style>body { font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #2d2d2d; color: white; }</style>')
                html_parts.append('</head><body>')
                
                html_parts.append(f'<h1>Question {q_idx} - Token Importance Heatmap</h1>')
                html_parts.append(f'<p><strong>Model:</strong> {model_name}</p>')
                html_parts.append(f'<p><strong>Press:</strong> {target_press} | <strong>Budget:</strong> {target_budget}</p>')
                html_parts.append(f'<p><strong>Compression events:</strong> {len(steps_with_scores)}</p>')
                html_parts.append(f'<p><strong>Question:</strong> {input_text[:300]}{"..." if len(input_text) > 300 else ""}</p>')
                
                html_parts.append(create_full_sequence_heatmap(steps_with_scores, all_token_ids, tokenizer, input_token_count or 0))
                
                html_parts.append('</body></html>')
                
                # Save HTML with descriptive name (use actual press from data)
                actual_press = question_data.get('press_name', target_press)
                actual_budget = question_data.get('cache_budget', target_budget)
                model_safe = target_model.replace("/", "_").replace("--", "_")
                html_path = output_dir / f"{actual_press}_budget{actual_budget}_q{q_idx}_{model_safe}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html_parts))
                
                print(f"  📄 Q{q_idx}: {len(all_token_ids)} tokens, {len(steps_with_scores)} compressions → {html_path.name}")
    
    if not matched_any:
        print("\n⚠️  No matching files found! Check your filters.")
    else:
        print("\n✅ Text heatmap visualization complete!")


if __name__ == "__main__":
    main()

