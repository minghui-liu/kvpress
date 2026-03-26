"""
GPQA (Graduate-Level Google-Proof Q&A) Dataset Handler
Supports both gpqa and gpqa_diamond variants

GPQA is a challenging dataset of multiple-choice questions in:
- Physics
- Chemistry
- Biology

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
Diamond subset: https://huggingface.co/datasets/Idavidrein/gpqa (config: gpqa_diamond)

Format:
- Question: Text question with multiple choice options
- Choices: List of answer choices (typically A, B, C, D)
- Answer: Index or letter of correct answer
"""

import re
from typing import Any, Dict, Optional, Tuple

# Prompt for GPQA multiple choice questions
gpqa_prompt = """

Please solve this graduate-level science question step by step.

Instructions:
1. Analyze the question carefully
2. Consider each option systematically
3. Explain your reasoning
4. Provide your final answer as a single letter (A, B, C, or D)
5. Wrap your final answer letter in \\boxed{}, for example: \\boxed{A}
"""


def _extract_gpqa_letter(text: str) -> Optional[str]:
    """
    Extract a multiple-choice letter (A/B/C/D) from model text.
    """
    if not text:
        return None

    boxed_match = re.search(r"\\boxed\{([A-D])\}", text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()

    search_text = text[-200:] if len(text) > 200 else text
    patterns = [
        r"(?:answer|choice|option)\s+(?:is\s+)?([A-D])",
        r"\b([A-D])\s+is\s+(?:correct|right)",
        r"(?:select|choose)\s+([A-D])",
        r"\(([A-D])\)",
        r"^([A-D])$",
    ]

    for pattern in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    letters = re.findall(r"\b([A-D])\b", search_text, re.IGNORECASE)
    if letters:
        return letters[-1].upper()

    return None


def gpqa_extractor(response: str) -> str:
    """
    Extract a GPQA prediction in a scorer-friendly format.

    Returns the predicted letter when available; otherwise returns the
    stripped response so evaluator fallback logic can still attempt parsing.
    """
    pred_letter = _extract_gpqa_letter(response)
    if pred_letter is not None:
        return pred_letter
    return response.strip() if response else ""


def gpqa_formatter(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Format example from GPQA dataset.

    GPQA datasets typically have these fields:
    - Question: The question text
    - choices: List of answer choices OR separate fields (A, B, C, D)
    - Answer: The correct answer (as index or letter)

    Args:
        example: Dataset example dict

    Returns:
        Tuple of (formatted_question, correct_answer)
    """
    # Extract question text - try multiple field name variations
    question_text = (
        example.get("Question")
        or example.get("question")
        or example.get("Problem")
        or example.get("problem")
        or example.get("query")
        or ""
    )
    if not question_text:
        # Provide helpful error message with available fields
        available_fields = list(example.keys())
        raise ValueError(
            f"Example missing question field. Available fields: {available_fields}"
        )

    # Format multiple choice options
    # GPQA datasets may have different formats for choices
    choices = None
    answer_letter = None

    # Try different field name patterns
    if "choices" in example:
        choices = example["choices"]
    elif "Choices" in example:
        choices = example["Choices"]
    elif all(key in example for key in ["A", "B", "C", "D"]):
        # Choices stored as separate fields
        choices = [
            example["A"],
            example["B"],
            example["C"],
            example["D"],
        ]
    elif all(key in example for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]):
        # GPQA format: Correct + 3 Incorrect answers
        # Shuffle them deterministically to create A/B/C/D options
        import random

        correct_ans = example["Correct Answer"]
        incorrect_answers = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        # Combine all answers
        all_answers = [correct_ans] + incorrect_answers

        # Create deterministic shuffle based on question text
        shuffled_indices = list(range(4))
        rng = random.Random(hash(question_text))
        rng.shuffle(shuffled_indices)

        # Apply shuffle
        choices = [all_answers[i] for i in shuffled_indices]

        # Find where the correct answer ended up
        answer_letter = chr(65 + shuffled_indices.index(0))

    # Build formatted question with choices
    formatted_question = question_text.strip()

    if choices:
        formatted_question += "\n\nOptions:"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            formatted_question += f"\n{letter}. {choice}"

    # Add prompt
    formatted_question += gpqa_prompt

    # Determine correct answer letter
    if answer_letter is None:
        # Extract correct answer from Answer field
        # Answer might be stored as letter (A/B/C/D) or index (0/1/2/3)
        answer = example.get("Answer") or example.get("answer", "")

        # Normalize answer to letter format
        if isinstance(answer, int):
            # If answer is index, convert to letter
            answer_letter = chr(65 + answer)  # 0->A, 1->B, etc
        elif isinstance(answer, str):
            # If already a letter, keep it; if it's a number string, convert it
            answer = answer.strip().upper()
            if answer.isdigit():
                answer_letter = chr(65 + int(answer))
            elif len(answer) == 1 and answer in "ABCD":
                answer_letter = answer
            else:
                # If answer is the full text, try to match it to choices
                if choices:
                    for i, choice in enumerate(choices):
                        if answer.lower() == choice.lower():
                            answer_letter = chr(65 + i)
                            break
                    else:
                        # Default to A if we can't match
                        answer_letter = "A"
                else:
                    answer_letter = "A"
        else:
            answer_letter = "A"  # Default fallback

    return formatted_question, answer_letter


def gpqa_evaluator(
    prediction: str,
    ground_truth: str,
    dataset_name: str = None,
    problem_text: str = None,
) -> bool:
    """
    Evaluate GPQA prediction against ground truth.

    For multiple choice questions, we check if the predicted letter (A/B/C/D)
    matches the ground truth letter.

    Args:
        prediction: Model's predicted answer
        ground_truth: Correct answer (letter A/B/C/D)
        dataset_name: Name of dataset (for context)
        problem_text: Original problem text (unused, for API compatibility)

    Returns:
        True if prediction matches ground truth, False otherwise
    """
    if prediction is None or ground_truth is None:
        return False

    prediction_text = str(prediction)
    ground_truth_text = str(ground_truth)

    pred_letter = _extract_gpqa_letter(prediction_text)
    if pred_letter is None:
        candidate = prediction_text.strip().upper()
        if len(candidate) == 1 and candidate in "ABCD":
            pred_letter = candidate
        else:
            return False

    gt_letter = ground_truth_text.strip().upper()
    if gt_letter.isdigit() and 0 <= int(gt_letter) <= 3:
        gt_letter = chr(65 + int(gt_letter))
    elif len(gt_letter) == 1 and gt_letter in "ABCD":
        pass
    else:
        gt_candidate = _extract_gpqa_letter(ground_truth_text)
        if gt_candidate is not None:
            gt_letter = gt_candidate
        else:
            gt_match = re.search(r"([A-D])", gt_letter, re.IGNORECASE)
            if gt_match:
                gt_letter = gt_match.group(1).upper()
            else:
                return False

    # Compare
    return pred_letter == gt_letter


def gpqa_scorer(predictions: list, answers: list) -> Dict[str, float]:
    """
    Score multiple GPQA predictions.

    Args:
        predictions: List of predicted answers
        answers: List of ground truth answers

    Returns:
        Dictionary with accuracy score
    """
    if len(predictions) != len(answers):
        raise ValueError(
            f"Predictions ({len(predictions)}) and answers ({len(answers)}) must have same length"
        )

    correct = sum(
        1 for pred, ans in zip(predictions, answers)
        if gpqa_evaluator(pred, ans)
    )

    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
    }
