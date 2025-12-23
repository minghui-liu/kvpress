"""
Check if the number after #### in the response matches the ground truth answer.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional


def extract_number_after_hash(response: str) -> Optional[str]:
    """
    Extract the number that appears after #### in the response.
    Returns None if no #### pattern is found.
    """
    if not response:
        return None
    
    # Look for #### followed by optional whitespace and then a number
    # Pattern: #### followed by whitespace and then digits (possibly with decimals)
    pattern = r'####\s*(\d+(?:\.\d+)?)'
    match = re.search(pattern, response)
    
    if match:
        return match.group(1).strip()
    
    return None


def check_answer_match(jsonl_file: Path, output_file: Optional[Path] = None):
    """
    Check each question in the JSONL file to see if the number after ####
    in the response matches the ground truth answer.
    
    Args:
        jsonl_file: Path to the JSONL file
        output_file: Optional path to save results (default: print to console)
    """
    results = []
    total = 0
    matched = 0
    no_hash_pattern = 0
    errors = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total += 1
                
                question = data.get('question', '')
                gt_answer = data.get('gt_answer', '')
                response = data.get('response', '')
                
                # Extract number after #### from response
                extracted_number = extract_number_after_hash(response)
                
                if extracted_number is None:
                    no_hash_pattern += 1
                    match_status = "NO_PATTERN"
                    is_match = False
                else:
                    # Compare with ground truth (both as strings, case-insensitive)
                    is_match = extracted_number.strip() == gt_answer.strip()
                    match_status = "MATCH" if is_match else "MISMATCH"
                    if is_match:
                        matched += 1
                
                result = {
                    'line': line_num,
                    'question': question[:100] + '...' if len(question) > 100 else question,
                    'gt_answer': gt_answer,
                    'extracted_number': extracted_number,
                    'status': match_status,
                    'match': is_match
                }
                results.append(result)
                
            except json.JSONDecodeError as e:
                errors += 1
                print(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                errors += 1
                print(f"Error processing line {line_num}: {e}")
    
    # Calculate statistics
    accuracy = (matched / total * 100) if total > 0 else 0.0
    
    # Prepare summary
    summary = {
        'total_questions': total,
        'matched': matched,
        'mismatched': total - matched - no_hash_pattern,
        'no_hash_pattern': no_hash_pattern,
        'errors': errors,
        'accuracy': accuracy
    }
    
    # Output results
    if output_file:
        output_data = {
            'summary': summary,
            'results': results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    else:
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"Summary for: {jsonl_file.name}")
        print(f"{'='*60}")
        print(f"Total questions: {total}")
        print(f"Matched: {matched}")
        print(f"Mismatched: {total - matched - no_hash_pattern}")
        print(f"No #### pattern found: {no_hash_pattern}")
        print(f"Errors: {errors}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}\n")
        
        # Print first few mismatches
        mismatches = [r for r in results if r['status'] == 'MISMATCH']
        if mismatches:
            print(f"\nFirst 5 mismatches:")
            for r in mismatches[:5]:
                print(f"  Line {r['line']}: GT={r['gt_answer']}, Extracted={r['extracted_number']}")
        
        # Print first few with no pattern
        no_pattern = [r for r in results if r['status'] == 'NO_PATTERN']
        if no_pattern:
            print(f"\nFirst 5 with no #### pattern:")
            for r in no_pattern[:5]:
                print(f"  Line {r['line']}: GT={r['gt_answer']}, Response preview: {response[:50] if 'response' in locals() else 'N/A'}...")
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(
        description="Check if numbers after #### in responses match ground truth"
    )
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to JSONL file to check"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON file to save detailed results"
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed results for each question"
    )
    
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        print(f"Error: File '{jsonl_path}' does not exist")
        return
    
    output_path = Path(args.output) if args.output else None
    
    summary, results = check_answer_match(jsonl_path, output_path)
    
    if args.show_details:
        print("\nDetailed results:")
        print("-" * 60)
        for r in results:
            status_symbol = "✓" if r['match'] else ("✗" if r['status'] == 'MISMATCH' else "?")
            print(f"{status_symbol} Line {r['line']:3d}: GT={r['gt_answer']:>6s}, "
                  f"Extracted={str(r['extracted_number']):>6s if r['extracted_number'] else 'N/A':>6s}, "
                  f"Status={r['status']}")


if __name__ == "__main__":
    main()

