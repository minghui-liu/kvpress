
openbookqa_prompt = "Solve the problem step by step. Answer with the answer choice label and wrap your final answer in \"\\boxed{}\"."


def openbookqa_formatter(example):
    """
    Format the example for openbookqa dataset.
    """
    choice_texts = example['choices']['text']
    choice_labels = example['choices']['label']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])

    input_text = f"Question:\n{example['question_stem']}\nChoices:\n{choices_text}\n{openbookqa_prompt}"
    # parse four # signs and the following text as the answer
    answer_text = example["answerKey"]

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
    

def openbookqa_scorer(predictions, answers):
    """
    Score the prediction for openbookqa dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict