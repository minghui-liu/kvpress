
from answer_grading import accuracy_from_comparator, choice_answers_equal


logiqa_prompt = "Given a context, a query and some options, pick the correct option."
logiqa_answer_prefix = "Solve the problem step by step. Answer with the option number and wrap your final answer in \"\\boxed{{}}\"."


def logiqa_formatter(example):
    """
    Format the example for logiqa dataset.
    """
    options_text = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(example["options"])])

    input_text = f"{logiqa_prompt}\nContext:\n{example['context']}\nQuery:\n{example['query']}\nOptions:\n{options_text}\n{logiqa_answer_prefix}"
    answer_text = str(example["correct_option"] + 1)  # Convert to 1-based index

    return input_text, answer_text


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    return accuracy_from_comparator(
        predictions, answers, lambda prediction, answer: choice_answers_equal(prediction, answer, max_choices=4)
    )
    

def logiqa_scorer(predictions, answers):
    """
    Score the prediction for logiqa dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict
