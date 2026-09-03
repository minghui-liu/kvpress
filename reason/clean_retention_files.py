"""
Clean retention visualization files.
Converts LaTeX and markdown to plain text while preserving **text** for bold rendering.
"""

import re
import html
from pathlib import Path
from typing import Optional


def clean_to_plain_text(text: str) -> str:
    """
    Convert LaTeX and markdown to plain text, but preserve **text** for bold rendering.
    Escapes markdown syntax so it appears as plain text, but keeps **text** intact.
    """
    # First, protect **text** markers by temporarily replacing them
    bold_pattern = r'\*\*([^*]+)\*\*'
    bold_markers = []
    counter = 0
    
    def replace_bold(match):
        nonlocal counter
        bold_markers.append(match.group(0))
        placeholder = f"__BOLD_MARKER_{counter}__"
        counter += 1
        return placeholder
    
    # Protect bold markers
    protected_text = re.sub(bold_pattern, replace_bold, text)
    
    # Escape markdown headers (##, ###, etc.) - add backslash to prevent rendering
    protected_text = re.sub(r'^(#{1,6})\s', r'\\\1 ', protected_text, flags=re.MULTILINE)
    
    # Remove LaTeX math delimiters ($$ and $)
    protected_text = re.sub(r'\$\$', '', protected_text)
    protected_text = re.sub(r'\$', '', protected_text)
    
    # Remove LaTeX commands but keep their content
    # \text{...} -> content only
    protected_text = re.sub(r'\\text\{([^}]*)\}', r'\1', protected_text)
    # \boxed{...} -> content only
    protected_text = re.sub(r'\\boxed\{([^}]*)\}', r'\1', protected_text)
    # \mathrm{...} -> content only
    protected_text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', protected_text)
    # \textbf{...} -> content only
    protected_text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', protected_text)
    
    # Remove LaTeX environments (array, aligned, etc.) - replace with newlines
    protected_text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', '\n', protected_text, flags=re.DOTALL)
    
    # Remove remaining LaTeX commands (like \hspace, \biggl, etc.)
    protected_text = re.sub(r'\\[a-zA-Z]+\*?', '', protected_text)
    
    # Clean up LaTeX braces - remove empty ones, keep content of others
    # This is a simple approach - remove braces around simple content
    protected_text = re.sub(r'\{([^{}]*)\}', r'\1', protected_text)
    
    # Decode HTML entities
    protected_text = html.unescape(protected_text)
    
    # Clean up extra whitespace
    protected_text = re.sub(r'\n{3,}', '\n\n', protected_text)
    protected_text = re.sub(r'[ \t]+', ' ', protected_text)
    
    # Restore bold markers
    for i, marker in enumerate(bold_markers):
        protected_text = protected_text.replace(f"__BOLD_MARKER_{i}__", marker)
    
    return protected_text.strip()


def process_file(input_path: Path, output_dir: Optional[Path] = None, output_ext: str = '.md') -> Path:
    """
    Process a single file: clean LaTeX/markdown while preserving **text**.
    
    Args:
        input_path: Path to input file
        output_dir: Directory to save output (default: same as input)
        output_ext: Output file extension (default: '.md')
    
    Returns:
        Path to output file
    """
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into answer and question sections
    if content.startswith('answer='):
        parts = content.split('\n\nquestion=', 1)
        if len(parts) == 2:
            answer_section = parts[0].replace('answer=', '', 1)
            question_section = parts[1] if parts[1] else ''
        else:
            answer_section = parts[0].replace('answer=', '', 1)
            question_section = ''
    else:
        # If no answer= prefix, treat entire content as answer
        answer_section = content
        question_section = ''
    
    # Clean both sections
    cleaned_answer = clean_to_plain_text(answer_section)
    cleaned_question = clean_to_plain_text(question_section) if question_section else ''
    
    # Reconstruct content
    if cleaned_question:
        cleaned_content = f"answer={cleaned_answer}\n\nquestion={cleaned_question}\n"
    else:
        cleaned_content = f"answer={cleaned_answer}\n"
    
    # Determine output path
    if output_dir is None:
        output_dir = input_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}{output_ext}"
    
    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean retention visualization files")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="retention_visualizations",
        help="Directory containing input files (default: retention_visualizations)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output files (default: same as input_dir)"
    )
    parser.add_argument(
        "--input_ext",
        type=str,
        default=".txt",
        help="Input file extension (default: .txt)"
    )
    parser.add_argument(
        "--output_ext",
        type=str,
        default=".md",
        help="Output file extension (default: .md)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional pattern to match specific files (e.g., '*h2o*')"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # Find all matching files
    if args.pattern:
        pattern = f"*{args.pattern}*{args.input_ext}"
        files = list(input_dir.glob(pattern))
    else:
        files = list(input_dir.glob(f"*{args.input_ext}"))
    
    if not files:
        print(f"No files found matching pattern '*{args.pattern or ''}*{args.input_ext}' in {input_dir}")
        return
    
    print(f"Found {len(files)} file(s) to process")
    
    for input_file in files:
        try:
            output_path = process_file(input_file, output_dir, args.output_ext)
            print(f"✅ Processed: {input_file.name} -> {output_path.name}")
        except Exception as e:
            print(f"❌ Error processing {input_file.name}: {e}")
    
    print(f"\n✅ Processing complete! Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()

