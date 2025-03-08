import re
import pandas as pd

def mt_string_match_part(preds, refs):
    """
    Calculate the string match score for all references
    preds: list of list of strings, shape (N, M) where N is the number of samples and M is the number of questions per sample
    refs: list of list of of list of strings, shape (N, M, K) where N is the number of samples, M is the number of questions per sample, and K is the number of references per question
    """
    scores = []
    for pred, ref in zip(preds, refs):
        for p, r in zip(pred, ref):
            score = sum([1.0 if v.lower() in p.lower() else 0.0 for v in r]) / len(r) * 100
            scores.append(score)
    return round(sum(scores) / len(scores), 2)

def mt_passage_scorer(preds, refs):
    """
    Calculate the string match score for all references
    preds: list of strings, predictions, shape (N,)
    refs: list of strings, references, shape (N,)
    """
    score = (
        sum(
            [sum([1.0 if r.lower() in p.lower() else 0.0 for p, r in zip(pred, ref)]) / len(ref) for pred, ref in zip(preds, refs)]
        )
        / len(preds)
        * 100
    )
    return round(score, 2)



METRICS_DICT = {
    "mt_niah": mt_string_match_part,
    "mt_vt": mt_string_match_part,
    "mt_passage": mt_passage_scorer,
}

def calculate_metrics(df: pd.DataFrame) -> dict:
    scores = {}
    # np_pattern = re.compile(r"[\x00-\x1f]")
    # df["predicted_answer"] = df["predicted_answer"].apply(lambda lst: [np_pattern.sub("", x.strip()).strip() for x in lst])
    df["predicted_answer"] = df["predicted_answer"].apply(lambda lst: [x.strip() for x in lst])

    metric_fn = METRICS_DICT[df["task"].iloc[0]]
    n_questions = len(df["questions"].iloc[0])
    for i in range(n_questions):
        preds = df["predicted_answer"].apply(lambda x: [x[i]]).tolist()
        refs = df["answers"].apply(lambda x: [x[i]]).tolist()
        score = metric_fn(preds, refs)
        scores[f"question_{i+1}"] = {"string_match": score}
    
    preds = df["predicted_answer"].tolist()
    refs = df["answers"].tolist()
    score = metric_fn(preds, refs)
    scores["all"] = {"string_match": score}

    return scores
