"""Shared, benchmark-aware answer extraction and grading helpers.

Math answers cannot be graded reliably with a single regular expression.  This
module uses exact rational arithmetic for numeric benchmarks and Math-Verify's
LaTeX/SymPy pipeline for MATH-style symbolic answers.  Classification tasks use
strict label parsing so that explanatory text cannot accidentally match a gold
label.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Callable, Iterable, Optional


_ANSWER_PREFIX_RE = re.compile(
    r"(?im)\b(?:the\s+)?(?:final\s+answer|correct\s+(?:answer|choice|option)|answer|choice|option)"
    r"\s*(?:is|should\s+be|:|=)\s*([^\n]+)"
)
_HASH_ANSWER_RE = re.compile(r"(?im)####\s*([^\n]+)")
_NUMBER_TOKEN_RE = re.compile(
    r"(?<![\w.])(?:[$€£¥₹₽₪₩₫฿]\s*)?"
    r"[+\-−]?(?:(?:\d[\d, ]*\.?\d*)|(?:\.\d+))"
    r"(?:[eE][+\-]?\d+)?(?:\s*/\s*[+\-−]?(?:(?:\d[\d, ]*\.?\d*)|(?:\.\d+))"
    r"(?:[eE][+\-]?\d+)?)?\s*%?"
)
_TRAILING_PUNCTUATION_RE = re.compile(r"[.。]\s*$")


def extract_boxed_content(text: object) -> list[str]:
    """Return all balanced ``\\boxed{...}`` payloads, including nested braces."""
    source = "" if text is None else str(text)
    results: list[str] = []
    marker = r"\boxed{"
    start_at = 0
    while True:
        marker_at = source.find(marker, start_at)
        if marker_at < 0:
            break
        payload_start = marker_at + len(marker)
        depth = 0
        for index in range(payload_start, len(source)):
            character = source[index]
            if character == "{":
                depth += 1
            elif character == "}":
                if depth == 0:
                    results.append(source[payload_start:index])
                    start_at = index + 1
                    break
                depth -= 1
        else:
            break
    return results


def _unwrap_text_commands(text: str) -> str:
    value = text.strip()
    command_re = re.compile(r"^\\(?:text|textrm|mathrm|mathbf|operatorname)\{(.*)\}$", re.DOTALL)
    while True:
        match = command_re.fullmatch(value)
        if not match:
            return value
        value = match.group(1).strip()


def extract_final_answer(response: object) -> str:
    """Extract an explicitly marked final answer without corrupting decimals."""
    text = "" if response is None else str(response).strip()
    if not text:
        return ""

    boxed = extract_boxed_content(text)
    if boxed:
        return _unwrap_text_commands(boxed[-1])

    matches = list(_HASH_ANSWER_RE.finditer(text)) + list(_ANSWER_PREFIX_RE.finditer(text))
    if matches:
        match = max(matches, key=lambda item: item.start())
        candidate = match.group(1).strip().strip("`*_ ")
        return _TRAILING_PUNCTUATION_RE.sub("", candidate).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and re.fullmatch(r"(?:\(?[A-Ea-e1-5]\)?[.)]?|true|false|uncertain|unknown)", lines[-1]):
        return lines[-1]
    return text


def _numeric_candidates(answer: object) -> Iterable[str]:
    text = "" if answer is None else str(answer).strip()
    if not text:
        return []

    candidates: list[str] = []
    candidates.extend(reversed(extract_boxed_content(text)))
    explicit = list(_HASH_ANSWER_RE.finditer(text)) + list(_ANSWER_PREFIX_RE.finditer(text))
    candidates.extend(match.group(1) for match in sorted(explicit, key=lambda item: item.start(), reverse=True))
    candidates.append(text)
    candidates.extend(match.group(0) for match in reversed(list(_NUMBER_TOKEN_RE.finditer(text))))

    seen: set[str] = set()
    return [candidate for candidate in candidates if not (candidate in seen or seen.add(candidate))]


def parse_numeric_answer(answer: object) -> Optional[Fraction]:
    """Parse one complete numeric answer using exact decimal/rational arithmetic.

    Currency symbols and trailing units are ignored. Percentages retain their
    mathematical meaning. No Python ``eval`` or other executable parser is used.
    """
    if answer is None:
        return None
    value = unicodedata.normalize("NFKC", str(answer)).strip()
    if not value:
        return None

    boxed = extract_boxed_content(value)
    if boxed and value == rf"\boxed{{{boxed[-1]}}}":
        value = boxed[-1].strip()
    value = _unwrap_text_commands(value)
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\(?:,|!|;|:|quad|qquad)", "", value)
    value = value.strip("`* ")
    value = value.replace(r"\$", "").replace("$", "")
    value = re.sub(r"^[€£¥₹₽₪₩₫฿]", "", value).strip()
    value = re.sub(r"^(?:[A-Za-z][A-Za-z0-9_]*)\s*=\s*", "", value)

    latex_fraction = re.fullmatch(
        r"[+\-]?\\(?:d?frac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}(?:\s*\\?%\s*)?",
        value,
    )
    if latex_fraction:
        sign = -1 if value.startswith("-") else 1
        numerator = parse_numeric_answer(latex_fraction.group(1))
        denominator = parse_numeric_answer(latex_fraction.group(2))
        if numerator is None or denominator in (None, 0):
            return None
        result = sign * numerator / denominator
        return result / 100 if value.rstrip().endswith("%") else result

    # Units are metadata for answer matching. Keep exponent notation (1e3)
    # intact while removing ordinary suffixes such as "dollars" or "cm^2".
    value = re.sub(r"(?:\\(?:text|mathrm)\{[^{}]*\}|\s+[A-Za-z°][A-Za-z0-9°^²³ /-]*)\s*$", "", value)
    value = value.strip().strip(".,;:!?")
    is_percent = bool(re.search(r"\\?%\s*$", value))
    value = re.sub(r"\\?%\s*$", "", value).strip()
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = value.replace(" ", "")

    number = r"[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+\-]?\d+)?"
    try:
        if re.fullmatch(number, value):
            result = Fraction(Decimal(value))
        else:
            fraction_match = re.fullmatch(rf"({number})/({number})", value)
            if not fraction_match:
                return None
            denominator = Fraction(Decimal(fraction_match.group(2)))
            if denominator == 0:
                return None
            result = Fraction(Decimal(fraction_match.group(1))) / denominator
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return result / 100 if is_percent else result


def numeric_answers_equal(prediction: object, gold: object, *, require_integer: bool = False) -> bool:
    """Compare a generated numeric answer with a gold answer exactly."""
    gold_value = parse_numeric_answer(gold)
    if gold_value is None or (require_integer and gold_value.denominator != 1):
        return False
    for candidate in _numeric_candidates(prediction):
        predicted_value = parse_numeric_answer(candidate)
        if predicted_value is None:
            continue
        if require_integer and predicted_value.denominator != 1:
            continue
        if predicted_value == gold_value:
            return True
    return False


def math_answers_equal(prediction: object, gold: object) -> bool:
    """Compare MATH-style answers using Math-Verify and deterministic fallbacks."""
    if prediction is None or gold is None:
        return False
    if numeric_answers_equal(prediction, gold):
        return True

    def normalize_surface(answer: object) -> str:
        value = unicodedata.normalize("NFKC", extract_final_answer(answer)).strip()
        value = _TRAILING_PUNCTUATION_RE.sub("", value)
        value = value.replace(r"\left", "").replace(r"\right", "")
        value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        value = value.replace(r"\!", "").replace(r"\,", "")
        value = value.strip("$ ")
        return re.sub(r"\s+", "", value)

    if normalize_surface(prediction) == normalize_surface(gold):
        return True

    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except ImportError as error:  # pragma: no cover - dependency is declared by the project
        raise RuntimeError(
            "MATH grading requires math-verify. Install the project dependencies "
            "or run `pip install math-verify[antlr4_13_2]`."
        ) from error

    gold_text = str(gold).strip()
    prediction_text = str(prediction).strip()
    if not gold_text or not prediction_text:
        return False

    # MATH-500 stores bare LaTeX, while Math-Verify intentionally requires a
    # LaTeX environment. Predictions may be full chain-of-thought responses.
    gold_for_parser = gold_text if re.search(r"(?:\$|\\\[|\\\(|\\boxed\{)", gold_text) else f"${gold_text}$"
    gold_parsed = parse(
        gold_for_parser,
        extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback",
    )
    prediction_candidates = [prediction_text]
    extracted_prediction = extract_final_answer(prediction_text)
    if extracted_prediction != prediction_text or "\n" not in prediction_text:
        prediction_candidates.append(
            extracted_prediction
            if re.search(r"(?:\$|\\\[|\\\(|\\boxed\{)", extracted_prediction)
            else f"${extracted_prediction}$"
        )

    if not gold_parsed:
        return False
    for candidate in dict.fromkeys(prediction_candidates):
        prediction_parsed = parse(
            candidate,
            extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
            fallback_mode="no_fallback",
        )
        if prediction_parsed and verify(gold_parsed, prediction_parsed):
            return True
    return False


def _choice_index(answer: object, max_choices: int) -> Optional[int]:
    raw = "" if answer is None else unicodedata.normalize("NFKC", str(answer))
    explicit_matches = list(
        re.finditer(
            r"(?i)\b(?:final\s+answer|correct\s+(?:answer|choice|option)|answer|choice|option)"
            r"\s*(?:is|should\s+be|:|=)\s*\(?([a-e]|[1-5])\)?",
            raw,
        )
    )
    candidate = explicit_matches[-1].group(1) if explicit_matches else extract_final_answer(raw)
    boxed = extract_boxed_content(candidate)
    if boxed:
        candidate = boxed[-1]
    value = unicodedata.normalize("NFKC", candidate).strip().lower()
    value = _unwrap_text_commands(value).strip().strip("`*()[]{} ")
    value = re.sub(r"[.)\]:,;!?]+$", "", value).strip()
    anchored = re.search(r"(?:answer|choice|option)\s*(?:is|:|=)?\s*\(?([a-e]|[1-5])\)?", value)
    if anchored:
        value = anchored.group(1)
    if re.fullmatch(r"[a-e]", value):
        index = ord(value) - ord("a")
    elif re.fullmatch(r"[1-5]", value):
        index = int(value) - 1
    else:
        return None
    return index if 0 <= index < max_choices else None


def choice_answers_equal(prediction: object, gold: object, *, max_choices: int) -> bool:
    predicted_index = _choice_index(prediction, max_choices)
    gold_index = _choice_index(gold, max_choices)
    return predicted_index is not None and predicted_index == gold_index


_CATEGORY_ALIASES = {
    "true": "true",
    "yes": "true",
    "false": "false",
    "no": "false",
    "uncertain": "uncertain",
    "unknown": "uncertain",
    "undetermined": "uncertain",
}


def categorical_answers_equal(prediction: object, gold: object, *, allowed: set[str]) -> bool:
    def normalize(answer: object) -> Optional[str]:
        raw = "" if answer is None else str(answer)
        explicit_matches = list(
            re.finditer(
                r"(?i)\b(?:final\s+answer|correct\s+answer|answer)\s*"
                r"(?:is|should\s+be|:|=)\s*(true|false|yes|no|uncertain|unknown|undetermined)\b",
                raw,
            )
        )
        value = explicit_matches[-1].group(1) if explicit_matches else extract_final_answer(raw)
        value = value.lower().strip().strip("`*()[]{} .,:;!?")
        value = _unwrap_text_commands(value)
        label = _CATEGORY_ALIASES.get(value)
        return label if label in allowed else None

    predicted_label = normalize(prediction)
    gold_label = normalize(gold)
    return predicted_label is not None and predicted_label == gold_label


def accuracy_from_comparator(
    predictions: list,
    answers: list,
    comparator: Callable[[object, object], bool],
) -> float:
    if len(predictions) != len(answers):
        raise ValueError(f"Predictions ({len(predictions)}) and answers ({len(answers)}) must have the same length")
    if not predictions:
        return 0.0
    return sum(comparator(prediction, answer) for prediction, answer in zip(predictions, answers)) / len(predictions)
