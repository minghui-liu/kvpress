import re
import string

import numpy as np
from scipy.optimize import linear_sum_assignment


drop_prompt = 'Solve the problem step by step. Wrap your final answer in "\\boxed{}".'


def drop_formatter(example):
    """Format a DROP example while preserving its possibly multi-span answer."""
    input_text = f"Passage:\n{example['passage']}\nQuestion:\n{example['question']}\n{drop_prompt}"
    answer_spans = example["answers_spans"]["spans"]
    if not isinstance(answer_spans, list):
        answer_spans = [answer_spans]
    return input_text, [str(answer) for answer in answer_spans]


def _coerce_spans(answer):
    """Represent a single answer or a multi-span answer as a list of strings."""
    if isinstance(answer, (list, tuple)):
        return [str(span) for span in answer]
    return [str(answer)]


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize(text):
    """Apply the normalization used by the official DROP evaluator."""
    normalized_tokens = []
    for token in re.split(r" |－|-", str(text).lower()):
        if not _is_number(token):
            token = "".join(character for character in token if character not in string.punctuation)
        if _is_number(token):
            token = str(float(token))
        token = re.sub(r"\b(a|an|the)\b", " ", token)
        token = " ".join(token.split())
        if token:
            normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def _answer_to_bags(answer):
    normalized_spans = [_normalize(span) for span in _coerce_spans(answer)]
    return normalized_spans, [set(span.split()) for span in normalized_spans]


def _bag_f1(predicted, gold):
    gold_numbers = {token for token in gold if _is_number(token)}
    predicted_numbers = {token for token in predicted if _is_number(token)}
    if gold_numbers and not gold_numbers.intersection(predicted_numbers):
        return 0.0

    overlap = len(predicted.intersection(gold))
    precision = overlap / len(predicted) if predicted else 1.0
    recall = overlap / len(gold) if gold else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def drop_metrics(predicted, gold):
    """Return official-style DROP exact match and token-level F1 for one answer."""
    predicted_spans, predicted_bags = _answer_to_bags(predicted)
    gold_spans, gold_bags = _answer_to_bags(gold)

    exact_match = float(set(predicted_spans) == set(gold_spans) and len(predicted_spans) == len(gold_spans))

    scores = np.zeros((len(gold_bags), len(predicted_bags)))
    for gold_index, gold_bag in enumerate(gold_bags):
        for predicted_index, predicted_bag in enumerate(predicted_bags):
            scores[gold_index, predicted_index] = _bag_f1(predicted_bag, gold_bag)
    rows, columns = linear_sum_assignment(-scores)
    aligned_scores = np.zeros(max(len(gold_bags), len(predicted_bags)))
    for row, column in zip(rows, columns):
        aligned_scores[row] = scores[row, column]
    return exact_match, round(float(np.mean(aligned_scores)), 2)


def drop_scorer(predictions, answers):
    """Aggregate DROP EM and F1; keep ``accuracy`` as a compatibility alias for EM."""
    scores = [drop_metrics(prediction, answer) for prediction, answer in zip(predictions, answers)]
    if not scores:
        return {"accuracy": 0.0, "exact_match": 0.0, "f1": 0.0}
    exact_match = sum(score[0] for score in scores) / len(scores)
    f1 = sum(score[1] for score in scores) / len(scores)
    return {"accuracy": exact_match, "exact_match": exact_match, "f1": f1}
