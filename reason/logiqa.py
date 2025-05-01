import re
from utils import extract_full_boxed_content, is_number

logiqa_prompt = "Given a context, a query and some options, pick the correct option."
logiqa_answer_prefix = "Solve the problem step by step. Answer with the option number and wrap your final answer in \"\\boxed{{}}\"."


def logiqa_formatter(example):
    """
    Format the example for logiqa dataset.
    """
    options_text = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(example["options"])])

    input_text = f"{logiqa_prompt}\nContext:\n{example['context']}\nQuery:\n{example['query']}\nOptions:\n{options_text}\n{logiqa_answer_prefix}"
    # parse four # signs and the following text as the answer
    answer_text = str(example["correct_option"])

    return input_text, answer_text


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    correct = 0
    total = len(predictions)

    for prediction, answer in zip(predictions, answers):
        if prediction.lower() == answer.lower():
            correct += 1

    return correct / total if total > 0 else 0.0
    

def logiqa_scorer(predictions, answers):
    """
    Score the prediction for logiqa dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict