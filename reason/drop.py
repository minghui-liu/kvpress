
drop_prompt = "Solve the problem step by step. Wrap your final answer in \"\\boxed{}\"."


def drop_formatter(example):
    """
    Format the example for drop dataset.
    """
    input_text = f"Passage:\n{example['passage']}\nQuestion:\n{example['question']}\n{drop_prompt}"
    # parse four # signs and the following text as the answer
    answer_text = example["answers_spans"]['spans']
    if isinstance(answer_text, list):
        answer_text = str(answer_text)
    else:
        answer_text = str([answer_text])

    return input_text, answer_text


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    correct = 0
    total = len(predictions)
 
    for prediction, answer_list in zip(predictions, answers):
        answer_list = [answer.lower() for answer in answer_list]
        if prediction.lower() in answer_list:
            correct += 1

    return correct / total if total > 0 else 0.0
    

def drop_scorer(predictions, answers):
    """
    Score the prediction for drop dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict