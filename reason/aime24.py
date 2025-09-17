
aime24_prompt = "\nSolve the problem step by step. Wrap your final answer in \"\\boxed{}\"."


def aime24_formatter(example):
    """
    Format the example for AIME24 dataset.
    """
    question_text = example["problem"] + aime25_prompt
    answer_text = example["answer"]

    return question_text, answer_text


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    correct = 0
    total = len(predictions)
    
    for prediction, answer in zip(predictions, answers):
        if prediction == answer:
            correct += 1

    return correct / total if total > 0 else 0.0
    

def aime24_scorer(predictions, answers):
    """
    Score the prediction for AIME25 dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict
