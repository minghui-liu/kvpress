from answer_grading import accuracy_from_comparator, categorical_answers_equal, extract_final_answer

folio_prompt = "Given the premises and conclusion, determine whether the conclusion follows from the premises. Answer with True, False or Uncertain."
folio_answer_prefix = "Solve the problem step by step. Choose your final answer from True, False and Uncertain and wrap your final answer in \"\\boxed{}\"."

def folio_formatter(example):
    """
    Format the example for folio dataset.
    """
    input_text = f"{folio_prompt}\nPremises:\n{example['premises']}\nConclusion:\n{example['conclusion']}\n{folio_answer_prefix}"
    # parse four # signs and the following text as the answer
    answer_text = example["label"]

    return input_text, answer_text


def folio_extractor(response):
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
            prediction, answer, allowed={"true", "false", "uncertain"}
        ),
    )
    

def folio_scorer(predictions, answers):
    """
    Score the prediction for folio dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict
