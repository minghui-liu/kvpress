
reclor_prompt = "Given a context, a question and some options, pick the correct option."
reclor_answer_prefix = "Solve the problem step by step. Answer with the option number and wrap your final answer in \"\\boxed{{}}\"."


def reclor_formatter(example):
    """
    Format the example for reclor dataset.
    """
    options_text = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(example["answers"])])

    input_text = f"{reclor_prompt}\nContext:\n{example['context']}\nQuestion:\n{example['question']}\nOptions:\n{options_text}\n{reclor_answer_prefix}"
    answer_text = str(example["label"] + 1)  # Convert to 1-based index

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
    

def reclor_scorer(predictions, answers):
    """
    Score the prediction for reclor dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict