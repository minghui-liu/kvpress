"""
Simple script to read step tracking files and display retained tokens.
For each question, goes directly to the last step and calculates retained tokens.
Tracks critical tokens (names, quantities, objects) and their retention.
"""

import json
import argparse
from collections import Counter
from pathlib import Path
from keyword_tracker import extract_keywords, tokenize_keywords
from transformers import AutoTokenizer


def load_json_stream(filepath: Path):
    """Load JSON objects from a file that may contain multiple JSON objects."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    decoder = json.JSONDecoder()
    pos = 0
    
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        
        try:
            obj, idx = decoder.raw_decode(text[pos:])
            pos += idx
            yield obj
        except json.JSONDecodeError:
            pos += 1


def process_file(filepath: Path, tokenizer_cache: dict, output_lines: list):
    """Process a single step tracking file and extract retained tokens from the last step of each question."""
    questions = list(load_json_stream(filepath))
    
    if not questions:
        return None
    
    # Get model name from first question
    model_name = questions[0].get('model_name') if questions else None
    if not model_name:
        error_msg = "  Error: Could not determine model name"
        print(error_msg)
        output_lines.append(error_msg)
        return None
    
    # Load tokenizer
    if model_name not in tokenizer_cache:
        try:
            tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            error_msg = f"  Error loading tokenizer for {model_name}: {e}"
            print(error_msg)
            output_lines.append(error_msg)
            return None
    
    tokenizer = tokenizer_cache[model_name]
    
    total_retained_tokens = 0
    total_critical_retained_tokens = 0
    total_critical_tokens = 0
    total_steps = 0
    
    for question_idx, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        
        generation_steps = question.get('generation_steps', [])
        if not generation_steps:
            continue
        
        last_step = generation_steps[-1]
        if not isinstance(last_step, dict):
            continue
        
        # Get retained tokens
        retained_tokens_text = last_step.get('retained_tokens_text', [])
        retained_tokens = last_step.get('retained_tokens', [])
        
        # Count retained tokens
        if retained_tokens:
            num_retained = len(retained_tokens)
        elif retained_tokens_text:
            num_retained = len(retained_tokens_text)
        else:
            num_retained = 0
        total_retained_tokens += num_retained
        total_steps += 1
        
        # Extract and track critical tokens with frequency counting
        input_text = question.get('input_text', '')
        if input_text:
            keywords = extract_keywords(input_text)
            keyword_token_ids = tokenize_keywords(keywords, tokenizer)
            
            # Count frequency of each critical token ID in the input text
            # Tokenize the entire input text to count occurrences
            input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
            input_token_frequency = Counter(input_token_ids)
            
            # Get all critical token IDs and their frequencies
            all_critical_token_ids = set()
            critical_token_frequencies = {}  # token_id -> frequency
            for token_set in keyword_token_ids.values():
                all_critical_token_ids.update(token_set)
                # Count frequency of each critical token in input
                for token_id in token_set:
                    if token_id in input_token_frequency:
                        critical_token_frequencies[token_id] = input_token_frequency[token_id]
                    else:
                        # If not found, at least count as 1 (shouldn't happen, but safety)
                        critical_token_frequencies[token_id] = critical_token_frequencies.get(token_id, 0) + 1
            
            # Get retained token IDs with frequency
            retained_token_ids = []
            if retained_tokens:
                retained_token_ids = retained_tokens  # Keep as list to preserve frequency
            elif retained_tokens_text:
                # Encode retained tokens text to get IDs
                try:
                    for token_text in retained_tokens_text:
                        if token_text:
                            tokens = tokenizer.encode(token_text, add_special_tokens=False)
                            retained_token_ids.extend(tokens)
                except Exception:
                    pass
            
            retained_token_frequency = Counter(retained_token_ids)
            
            # Count frequency of retained critical tokens
            # For each critical token ID, count how many times it appears in retained tokens
            num_critical_retained = 0
            for token_id in all_critical_token_ids:
                if token_id in retained_token_frequency:
                    num_critical_retained += retained_token_frequency[token_id]
            
            # Count total frequency of critical tokens in input
            num_critical_total = sum(critical_token_frequencies.values())
            
            # Get the actual keyword strings that were retained (with frequency info)
            # A keyword is considered retained if at least one of its tokens is retained
            retained_critical_keywords = {}
            for key_type, keyword_list in keywords.items():
                retained_keywords = []
                for keyword in keyword_list:
                    keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
                    # Check if any token of this keyword is retained
                    if any(tid in retained_token_frequency for tid in keyword_tokens):
                        # Count how many times this keyword appears in input
                        keyword_freq = input_text.lower().count(keyword.lower())
                        if keyword_freq > 1:
                            retained_keywords.append(f"{keyword}({keyword_freq}x)")
                        else:
                            retained_keywords.append(keyword)
                retained_critical_keywords[key_type] = retained_keywords
            
            total_critical_retained_tokens += num_critical_retained
            total_critical_tokens += num_critical_total
            
            lines = [
                f"Question {question_idx}:",
                f"  Number of Retained Tokens: {num_retained}",
                f"  Critical Tokens Retained: {retained_critical_keywords}",
                f"  Number of Retained Critical Tokens: {num_critical_retained} / {num_critical_total}",
                ""
            ]
            for line in lines:
                print(line)
                output_lines.append(line)
        else:
            lines = [
                f"Question {question_idx}:",
                f"  Number of Retained Tokens: {num_retained}",
                "  Critical Tokens Retained: (no input text)",
                ""
            ]
            for line in lines:
                print(line)
                output_lines.append(line)
    
    avg_retained_per_question = total_retained_tokens / len(questions) if questions else 0
    avg_critical_retained_per_question = total_critical_retained_tokens / len(questions) if questions else 0
    avg_critical_total_per_question = total_critical_tokens / len(questions) if questions else 0
    
    lines = [
        "-" * 50,
        f"Average Retained Tokens per Question: {avg_retained_per_question:.2f}",
        f"Average Retained Critical Tokens per Question: {avg_critical_retained_per_question:.2f}",
        f"Average Total Critical Tokens per Question: {avg_critical_total_per_question:.2f}",
    ]
    if avg_critical_total_per_question > 0:
        lines.append(f"Critical Token Retention Rate: {(avg_critical_retained_per_question / avg_critical_total_per_question * 100):.2f}%")
    lines.append("-" * 50)
    
    for line in lines:
        print(line)
        output_lines.append(line)
    
    return {
        'avg_per_question': avg_retained_per_question,
        'avg_critical_per_question': avg_critical_retained_per_question,
        'total_questions': len(questions),
        'total_steps': total_steps
    }


def main():
    parser = argparse.ArgumentParser(
        description="View retained tokens from step tracking files."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="token_tracking_results",
        help="Directory containing .step_tracking.json files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="retention_results.txt",
        help="Output file to write results to (default: retention_results.txt)"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return
    
    step_tracking_files = list(results_dir.glob("*.step_tracking.json"))
    if not step_tracking_files:
        print(f"No .step_tracking.json files found in {results_dir}")
        return
    
    tokenizer_cache = {}
    output_lines = []
    
    # Add header
    header = [
        "=" * 70,
        "Token Retention Analysis Results",
        "=" * 70,
        ""
    ]
    for line in header:
        print(line)
        output_lines.append(line)
    
    for step_file in sorted(step_tracking_files):
        file_header = f"Processing: {step_file.name}..."
        print(file_header)
        output_lines.append(file_header)
        output_lines.append("")
        process_file(step_file, tokenizer_cache, output_lines)
        output_lines.append("")
        print()
    
    # Write all results to file
    output_path = Path(args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✅ All results written to: {output_path}")


if __name__ == "__main__":
    main()
