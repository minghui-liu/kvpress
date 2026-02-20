"""
IFEval dataset integration for kvpress evaluation framework.

IFEval (Instruction-Following Evaluation) tests whether models can follow
verifiable natural language instructions.
"""

import json
import re
from typing import List, Dict, Any


def ifeval_formatter(example):
    """
    Format the example for IFEval dataset.

    For IFEval, we just use the prompt as-is, and the "answer" is whether
    the generated response follows all the verifiable instructions.
    """
    prompt = example["prompt"]
    # For IFEval, there is no single correct answer - we score based on instruction following
    # We return None for the answer since it's computed by the scorer
    return prompt, None


def _check_punctuation_no_comma(response: str) -> bool:
    """Check that response contains no commas."""
    return ',' not in response


def _check_detectable_format_number_highlighted_sections(response: str, num_highlights: int) -> bool:
    """Check that response contains at least num_highlights markdown-style highlights."""
    # Look for patterns like *text* (markdown italics/bold)
    highlights = re.findall(r'\*[^*]+\*', response)
    return len(highlights) >= num_highlights


def _check_length_constraints_number_words(response: str, relation: str, num_words: int) -> bool:
    """Check word count constraints."""
    words = response.split()
    word_count = len(words)

    if relation == "at least":
        return word_count >= num_words
    elif relation == "at most":
        return word_count <= num_words
    elif relation == "exactly":
        return word_count == num_words
    else:
        # Default to at least
        return word_count >= num_words


def _check_length_constraints_number_sentences(response: str, relation: str, num_sentences: int) -> bool:
    """Check sentence count constraints."""
    # Simple sentence splitting on periods, question marks, exclamation marks
    sentences = re.split(r'[.!?]+', response.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    if relation == "at least":
        return sentence_count >= num_sentences
    elif relation == "at most":
        return sentence_count <= num_sentences
    elif relation == "exactly":
        return sentence_count == num_sentences
    elif relation == "less than":
        return sentence_count < num_sentences
    else:
        # Default to at least
        return sentence_count >= num_sentences


def _check_detectable_format_number_bullet_lists(response: str, num_bullets: int) -> bool:
    """Check that response contains at least num_bullets markdown bullet points."""
    # Look for markdown bullet points: - item, * item, or + item
    bullets = re.findall(r'^[ \t]*[-*+][ \t]+.*$', response, re.MULTILINE)
    return len(bullets) >= num_bullets


def _check_keywords_forbidden_words(response: str, forbidden_words: List[str]) -> bool:
    """Check that response does not contain any forbidden words."""
    if not forbidden_words:
        return True

    response_lower = response.lower()
    for word in forbidden_words:
        if word.lower() in response_lower:
            return False
    return True


def _check_detectable_content_number_placeholders(response: str, num_placeholders: int) -> bool:
    """Check that response contains at least num_placeholders square bracket placeholders."""
    # Look for [placeholder] patterns
    placeholders = re.findall(r'\[.*?\]', response)
    return len(placeholders) >= num_placeholders


def _check_detectable_content_postscript(response: str, postscript_marker: str) -> bool:
    """Check that response ends with a postscript marker."""
    if not postscript_marker:
        return True

    response_stripped = response.strip()
    return response_stripped.endswith(postscript_marker)


def _check_keywords_existence(response: str, keywords: List[str]) -> bool:
    """Check that response contains all specified keywords."""
    if not keywords:
        return True

    response_lower = response.lower()
    for keyword in keywords:
        if keyword.lower() not in response_lower:
            return False
    return True


def _check_detectable_format_json_format(response: str) -> bool:
    """Check that response is valid JSON format."""
    response_stripped = response.strip()
    if not response_stripped:
        return False

    try:
        json.loads(response_stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _check_startend_quotation(response: str) -> bool:
    """Check that response starts and ends with quotation marks."""
    response_stripped = response.strip()
    return response_stripped.startswith('"') and response_stripped.endswith('"')


def _check_detectable_format_multiple_sections(response: str, section_spliter: str, num_sections: int) -> bool:
    """Check that response contains specified number of sections."""
    if not section_spliter or num_sections <= 0:
        return True

    sections = response.split(section_spliter)
    # Filter out empty sections
    sections = [s.strip() for s in sections if s.strip()]
    return len(sections) >= num_sections


def _verify_instruction(response: str, instruction_id: str, kwargs: Dict[str, Any]) -> bool:
    """Verify a single instruction."""
    if instruction_id == "punctuation:no_comma":
        return _check_punctuation_no_comma(response)

    elif instruction_id == "detectable_format:number_highlighted_sections":
        num_highlights = kwargs.get("num_highlights", 0)
        return _check_detectable_format_number_highlighted_sections(response, num_highlights)

    elif instruction_id == "detectable_format:number_bullet_lists":
        num_bullets = kwargs.get("num_bullets", 0)
        return _check_detectable_format_number_bullet_lists(response, num_bullets)

    elif instruction_id == "length_constraints:number_words":
        relation = kwargs.get("relation", "at least")
        num_words = kwargs.get("num_words", 0)
        return _check_length_constraints_number_words(response, relation, num_words)

    elif instruction_id == "length_constraints:number_sentences":
        relation = kwargs.get("relation", "at least")
        num_sentences = kwargs.get("num_sentences", 0)
        return _check_length_constraints_number_sentences(response, relation, num_sentences)

    elif instruction_id == "keywords:forbidden_words":
        forbidden_words = kwargs.get("forbidden_words", [])
        return _check_keywords_forbidden_words(response, forbidden_words)

    elif instruction_id == "detectable_content:number_placeholders":
        num_placeholders = kwargs.get("num_placeholders", 0)
        return _check_detectable_content_number_placeholders(response, num_placeholders)

    elif instruction_id == "detectable_content:postscript":
        postscript_marker = kwargs.get("postscript_marker", "")
        return _check_detectable_content_postscript(response, postscript_marker)

    elif instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords", [])
        return _check_keywords_existence(response, keywords)

    elif instruction_id == "detectable_format:json_format":
        return _check_detectable_format_json_format(response)

    elif instruction_id == "startend:quotation":
        return _check_startend_quotation(response)

    elif instruction_id == "detectable_format:multiple_sections":
        section_spliter = kwargs.get("section_spliter", "")
        num_sections = kwargs.get("num_sections", 0)
        return _check_detectable_format_multiple_sections(response, section_spliter, num_sections)

    else:
        # Unknown instruction - default to True (don't penalize unknown instructions)
        print(f"Warning: Unknown instruction type: {instruction_id}")
        return True


def ifeval_scorer(predictions: List[str], answers: List[str], instruction_lists: List[List[str]], kwargs_lists: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Score IFEval predictions based on instruction following.

    Args:
        predictions: List of model responses
        answers: Ignored (IFEval has no single correct answer)
        instruction_lists: List of instruction_id_lists for each sample
        kwargs_lists: List of kwargs lists for each sample

    Returns:
        Dict with accuracy scores
    """
    if len(predictions) != len(instruction_lists) or len(predictions) != len(kwargs_lists):
        raise ValueError("Predictions, instruction_lists, and kwargs_lists must have same length")

    total_samples = len(predictions)
    correct_samples = 0

    for response, instruction_list, kwargs_list in zip(predictions, instruction_lists, kwargs_lists):
        # Check if ALL instructions are followed for this sample
        all_instructions_passed = True

        for instruction_id, kwargs in zip(instruction_list, kwargs_list):
            if not _verify_instruction(response, instruction_id, kwargs):
                all_instructions_passed = False
                break

        if all_instructions_passed:
            correct_samples += 1

    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct_samples": correct_samples,
        "total_samples": total_samples
    }