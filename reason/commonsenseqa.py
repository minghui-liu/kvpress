from answer_grading import accuracy_from_comparator, choice_answers_equal

commonsenseqa_prompt = "Solve the problem step by step. Answer with the answer choice label and wrap your final answer in \"\\boxed{}\"."

def commonsenseqa_formatter(example):
    """
    Format the example for commonsenseqa dataset.
    """
    choice_texts = example['choices']['text']
    choice_labels = example['choices']['label']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])

    input_text = f"Question:\n{example['question']}\nChoices:\n{choices_text}\n{commonsenseqa_prompt}"
    # parse four # signs and the following text as the answer
    answer_text = example["answerKey"]

    return input_text, answer_text


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    return accuracy_from_comparator(
        predictions, answers, lambda prediction, answer: choice_answers_equal(prediction, answer, max_choices=5)
    )
    

def commonsenseqa_scorer(predictions, answers):
    """
    Score the prediction for commonsenseqa dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict
