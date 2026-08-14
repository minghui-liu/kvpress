from answer_grading import accuracy_from_comparator, categorical_answers_equal, extract_final_answer

strategyqa_prompt = "Given some facts and a related question, answer the question with true or false."
strategyqa_answer_prefix = "Solve the problem step by step. Answer with true or false and wrap your final answer in \"\\boxed{}\"."


def strategyqa_formatter(example):
    """
    Format the example for strategyqa dataset.
    """
    input_text = (
        f"{strategyqa_prompt}\nFacts:\n{example['facts']}\n"
        f"Question:\n{example['question']}\n{strategyqa_answer_prefix}"
    )
    # parse four # signs and the following text as the answer
    answer_text = str(example["answer"])

    return input_text, answer_text


def strategyqa_extractor(response):
    """
    Parse the answer text to get the answer.
    """
    return extract_final_answer(response)


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    return accuracy_from_comparator(
        predictions,
        answers,
        lambda prediction, answer: categorical_answers_equal(
            prediction, answer, allowed={"true", "false"}
        ),
    )
    

def strategyqa_scorer(predictions, answers):
    """
    Score the prediction for strategy qa dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict
