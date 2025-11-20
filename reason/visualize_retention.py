"""
Visualize token retention in model answers.
For a given question, shows which tokens in the answer were retained in the KV cache.
"""

import argparse
import json
from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from keyword_tracker import extract_keywords
import re


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


def extract_all_important_tokens(text: str) -> List[str]:
    """
    Extract all important tokens for GSM8K problems including:
    - Names, quantities, objects (from keyword_tracker)
    - Mathematical operators and words
    - Units and measurements
    - Key problem-solving words
    - Important numbers and ratios
    """
    important_tokens = []
    
    # Get basic keywords (names, quantities, objects)
    keywords = extract_keywords(text)
    for key_type, keyword_list in keywords.items():
        important_tokens.extend(keyword_list)
    
    # Mathematical operators and words
    math_words = [
        r'\b(plus|minus|times|multiplied|divided|add|subtract|multiply|divide|sum|difference|product|quotient)\b',
        r'\b(equals?|is equal to|=\s*\d+)',
        r'\b(more than|less than|greater than|fewer than)',
        r'\b(per|each|every|total|together|combined)',
        r'\b(ratio|proportion|fraction|percent|percentage)',
    ]
    for pattern in math_words:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if isinstance(matches[0], tuple) if matches else False:
            matches = [m[0] for m in matches if m]
        important_tokens.extend([m.strip() for m in matches if m and isinstance(m, str)])
    
    # Units and measurements (more comprehensive)
    units = re.findall(r'\b(dollars?|cents?|hours?|minutes?|days?|weeks?|months?|years?|'
                      r'pounds?|ounces?|kilograms?|grams?|meters?|kilometers?|miles?|'
                      r'pieces?|items?|units?|boxes?|bags?|bottles?|cups?|plates?)\b', 
                      text, re.IGNORECASE)
    important_tokens.extend(units)
    
    # Key problem-solving words
    problem_words = re.findall(r'\b(calculate|find|determine|solve|compute|figure out|'
                              r'how many|how much|what is|what are|total|remaining|left|'
                              r'each|per|every|both|all|together|combined)\b', 
                              text, re.IGNORECASE)
    important_tokens.extend(problem_words)
    
    # Numbers in various formats (including ratios like 7:11)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)  # Regular numbers
    ratios = re.findall(r'\d+:\d+', text)  # Ratios
    percentages = re.findall(r'\d+(?:\.\d+)?%', text)  # Percentages
    important_tokens.extend(numbers)
    important_tokens.extend(ratios)
    important_tokens.extend(percentages)
    
    # Important time/age words
    time_words = re.findall(r'\b(age|ages|old|older|younger|ago|from now|later|before|after)\b', 
                           text, re.IGNORECASE)
    important_tokens.extend(time_words)
    
    # Important action verbs
    action_words = re.findall(r'\b(buy|sells?|purchases?|spends?|earns?|saves?|'
                             r'gives?|receives?|takes?|leaves?|has|have|gets?|'
                             r'installs?|monitors?|goes?|walks?|runs?|drives?)\b', 
                             text, re.IGNORECASE)
    important_tokens.extend(action_words)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in important_tokens:
        token_lower = token.lower().strip()
        if token_lower and token_lower not in seen and len(token.strip()) > 0:
            seen.add(token_lower)
            unique_tokens.append(token.strip())
    
    return unique_tokens


def main():
    parser = argparse.ArgumentParser(description="Visualize token retention in model answers")
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Cache budget (e.g., 256)"
    )
    parser.add_argument(
        "--press",
        type=str,
        required=True,
        help="Press name (e.g., 'full')"
    )
    parser.add_argument(
        "--question_index",
        type=int,
        default=5,
        help="Question index to visualize (default: 5)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "meta-llama--Meta-Llama-3.1-8B-Instruct",
            "deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"
        ],
        help="List of model names to process (default: meta-llama--Meta-Llama-3.1-8B-Instruct deepseek-ai--DeepSeek-R1-Distill-Qwen-7B)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="token_tracking_results",
        help="Directory containing step_tracking.json files (default: token_tracking_results)"
    )
    
    args = parser.parse_args()
    
    target_budget = args.budget
    target_press_name = args.press
    target_model_list = [
        "meta-llama--Meta-Llama-3.1-8B-Instruct",
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
        "nvidia--Llama-3.1-Nemotron-Nano-8B-v1",
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B",
    ]
    question_index = args.question_index
    
    # load all step_tracking files from token_tracking_results directory
    step_tracking_files = list(Path(args.results_dir).glob("*.step_tracking.json"))

    tokenizer_cache = {}

    for step_tracking_file in step_tracking_files:
        for target_model in target_model_list:
            # Fix the expected filename pattern
            expected_file_name = f"gsm8k__main__{target_model}__{target_press_name}__budget{target_budget}__max_new_tokens2048__num_samples15__sampling.step_tracking.json"
            
            if expected_file_name == step_tracking_file.name:
                print(f"Processing: {step_tracking_file.name}")
                
                # Load step tracking data
                questions = list(load_json_stream(step_tracking_file))
                
                if question_index >= len(questions):
                    print(f"  Question index {question_index} not found (only {len(questions)} questions)")
                    continue
                
                question_data = questions[question_index]
                
                # Get model name and load tokenizer
                model_name = question_data.get('model_name')
                if not model_name:
                    print("  No model name found")
                    continue
                
                if model_name not in tokenizer_cache:
                    try:
                        tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                            model_name, trust_remote_code=True
                        )
                        print(f"  Loaded tokenizer for {model_name}")
                    except Exception as e:  # noqa: E722
                        print(f"  Error loading tokenizer: {e}")
                        continue
                
                tokenizer = tokenizer_cache[model_name]
                
                # Find out the finally retained tokens in the last step of the question_index
                generation_steps = question_data.get('generation_steps', [])
                if not generation_steps:
                    print("  No generation steps found")
                    continue
                
                last_step = generation_steps[-1]
                retained_tokens_text = last_step.get('retained_tokens_text', [])
                retained_tokens = last_step.get('retained_tokens', [])
                
                # Get retained token IDs
                retained_token_ids = set()
                if retained_tokens:
                    retained_token_ids = set(retained_tokens)
                elif retained_tokens_text:
                    # Encode retained tokens text to get IDs
                    for token_text in retained_tokens_text:
                        if token_text:
                            try:
                                tokens = tokenizer.encode(token_text, add_special_tokens=False)
                                retained_token_ids.update(tokens)
                            except Exception:  # noqa: E722
                                pass
                
                print(f"  Found {len(retained_token_ids)} retained token IDs")
                
                # Go to the sampling file (jsonl)
                sampling_file_name = step_tracking_file.stem.replace('.step_tracking', '') + '.jsonl'
                sampling_file = step_tracking_file.parent / sampling_file_name
                
                if not sampling_file.exists():
                    print(f"  Sampling file not found: {sampling_file}")
                    continue
                
                # Find out the question and answer
                with open(sampling_file, 'r', encoding='utf-8') as sampling_f:
                    lines = sampling_f.readlines()
                    if question_index >= len(lines):
                        print(f"  Question index {question_index} not found in sampling file")
                        continue
                    
                    answer_data = json.loads(lines[question_index])
                    question_text = answer_data.get('input_text', '')
                    answer_text = answer_data.get('response', '')
                    
                    if not answer_text:
                        print("  No answer text found")
                        continue
                
                print(f"  Question: {question_text[:100]}...")
                print(f"  Answer length: {len(answer_text)} chars")
                
                # Print full answer
                print(f"\n  Full Answer:\n{answer_text}\n")
                
                # Extract all important tokens from question (not just names, quantities, objects)
                important_keywords = extract_all_important_tokens(question_text)
                
                # Tokenize important keywords to get token IDs
                all_critical_token_ids = set()
                critical_keyword_map = {}  # Map token_id -> keyword for highlighting
                
                for keyword in important_keywords:
                    try:
                        keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
                        for token_id in keyword_tokens:
                            all_critical_token_ids.add(token_id)
                            if token_id not in critical_keyword_map:
                                critical_keyword_map[token_id] = []
                            critical_keyword_map[token_id].append(keyword)
                    except Exception:  # noqa: E722
                        pass
                
                # Find which critical tokens are retained
                retained_critical_token_ids = all_critical_token_ids & retained_token_ids
                
                # Get the actual keywords that were retained
                retained_keywords = set()
                for token_id in retained_critical_token_ids:
                    if token_id in critical_keyword_map:
                        retained_keywords.update(critical_keyword_map[token_id])
                
                print(f"  Important tokens found: {len(all_critical_token_ids)}")
                print(f"  Important tokens retained: {len(retained_critical_token_ids)}")
                print(f"  Retained keywords: {sorted(retained_keywords)[:20]}...")  # Show first 20
                
                # Generate markdown file with bolded retained keywords
                def highlight_keywords_in_text(text: str, keywords: set) -> str:
                    """Highlight keywords in text using markdown bold syntax."""
                    if not keywords:
                        return text
                    
                    # Sort keywords by length (longest first) to avoid partial matches
                    sorted_keywords = sorted(keywords, key=len, reverse=True)
                    
                    # Find all matches with their positions
                    matches = []
                    text_lower = text.lower()
                    
                    for kw in sorted_keywords:
                        keyword_lower = kw.lower()
                        start = 0
                        while True:
                            idx = text_lower.find(keyword_lower, start)
                            if idx == -1:
                                break
                            matches.append((idx, idx + len(kw), kw))
                            start = idx + 1
                    
                    # Sort matches by position
                    matches.sort(key=lambda x: x[0])
                    
                    # Remove overlapping matches (keep longer ones)
                    non_overlapping = []
                    for match in matches:
                        start, end, kw = match
                        # Check if it overlaps with any existing match
                        overlaps = False
                        to_remove = []
                        for i, (existing_start, existing_end, _) in enumerate(non_overlapping):
                            if not (end <= existing_start or start >= existing_end):
                                overlaps = True
                                # If current is longer, mark existing for removal
                                if (end - start) > (existing_end - existing_start):
                                    to_remove.append(i)
                                break
                        # Remove marked items (in reverse order to maintain indices)
                        for i in reversed(to_remove):
                            non_overlapping.pop(i)
                        if not overlaps or to_remove:
                            non_overlapping.append(match)
                    
                    # Sort again after potential removals
                    non_overlapping.sort(key=lambda x: x[0])
                    
                    # Build result by inserting bold markers
                    result_parts = []
                    last_pos = 0
                    
                    for start, end, kw in non_overlapping:
                        # Add text before this match
                        if start > last_pos:
                            result_parts.append(text[last_pos:start])
                        # Add the highlighted keyword (preserve original case)
                        actual_keyword = text[start:end]
                        result_parts.append(f"**{actual_keyword}**")
                        last_pos = end
                    
                    # Add remaining text
                    if last_pos < len(text):
                        result_parts.append(text[last_pos:])
                    
                    return ''.join(result_parts)
                
                # Generate TXT content
                # Format: answer= <answer with bold keywords>, then question= <question>
                highlighted_answer = highlight_keywords_in_text(answer_text, retained_keywords)
                highlighted_question = highlight_keywords_in_text(question_text, retained_keywords)
                
                txt_content = f"answer={highlighted_answer}\n\nquestion={highlighted_question}\n"
                
                # Save TXT file - each model gets its own separate file
                output_dir = Path("retention_visualizations")
                output_dir.mkdir(exist_ok=True)
                
                # Create unique filename for each model
                model_safe_name = model_name.replace('/', '_').replace('-', '_')
                output_path = output_dir / f"question_{question_index}_{target_press_name}_budget{target_budget}_{model_safe_name}.txt"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(txt_content)
                
                print(f"  ✅ Saved TXT to: {output_path}\n")

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
