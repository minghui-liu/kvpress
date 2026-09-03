
from answer_grading import accuracy_from_comparator, numeric_answers_equal


gsm8k_prompt = "\nSolve the problem step by step. Wrap your final answer in \"\\boxed{}\"."


def gsm8k_formatter(example):
    """
    Format the example for GSM8K dataset.
    """
    question_text = example["question"] + gsm8k_prompt
    # parse four # signs and the following text as the answer
    answer_text = example["answer"].split("####")[-1].strip()

    return question_text, answer_text


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    return accuracy_from_comparator(predictions, answers, numeric_answers_equal)
    

def gsm8k_scorer(predictions, answers):
    """
    Score the prediction for GSM8K dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict
