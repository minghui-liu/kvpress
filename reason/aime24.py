
from answer_grading import accuracy_from_comparator, extract_boxed_content, numeric_answers_equal


aime24_prompt = "\nSolve the problem step by step. Wrap your final answer in \"\\boxed{}\"."


def aime24_formatter(example):
    """
    Format the example for AIME24 dataset.
    """
    question_text = example["problem"] + aime24_prompt
    boxed_answers = extract_boxed_content(example["solution"])
    if not boxed_answers:
        raise ValueError("AIME24 solution does not contain a boxed gold answer")
    answer_text = boxed_answers[-1].strip()

    return question_text, answer_text


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    return accuracy_from_comparator(
        predictions,
        answers,
        lambda prediction, answer: numeric_answers_equal(prediction, answer, require_integer=True),
    )
    

def aime24_scorer(predictions, answers):
    """
    Score the prediction for AIME24 dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict
