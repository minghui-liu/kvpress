import operator
import re
import pandas as pd


def mt_string_match(preds, refs):
    """
    Calculate the string match score for all references
    preds: list of strings, shape (N,) where N is the number of samples
    refs: list of strings, references, shape (N,) OR list of list of strings, shape (N, K) where N is the number of samples and K is the number of references per question
    """
    scores = []
    for pred, ref in zip(preds, refs):
        pred = pred.lower()
        if isinstance(ref, list):
            score = sum([1.0 if r.lower() in pred else 0.0 for r in ref]) / len(ref) * 100
        else:
            ref = ref.lower()
            score = 100.0 if ref in pred else 0.0
        scores.append(score)
    return sum(scores) / len(scores)

def calculate_metrics(predictions: list, references: list) -> dict:
    scores = {} 
    metric_fn = mt_string_match
    n_questions = len(predictions[0])
    for i in range(n_questions):
        preds = list(map(operator.itemgetter(i), predictions))
        refs = list(map(operator.itemgetter(i), references))
        score = metric_fn(preds, refs)
        scores[f"question_{i+1}"] = {"string_match": score}
    
    avg_score = sum([score["string_match"] for score in scores.values()]) / n_questions
    scores["all"] = {"string_match": avg_score}

    return scores
