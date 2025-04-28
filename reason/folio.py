import re
from utils import extract_full_boxed_content, is_number

folio_prompt = "Given the premises and conclusion, determine whether the conclusion follows from the premises. Answer with True, False or Uncertain."
folio_answer_prefix = "Solve the problem step by step. Choose your final answer from True, False and Uncertain and wrap your final answer in one \"\\boxed{{}}\"."
# folio_answer_prefix = "Choose your final answer from True, False and Uncertain and wrap your final answer in one \"\\boxed{{}}\"."


def folio_formatter(example):
    """
    Format the example for folio dataset.
    """
    input_text = f"{folio_prompt}\nPremises:\n{example['premises']}\nConclusion:\n{example['conclusion']}\n{folio_answer_prefix}"
    # parse four # signs and the following text as the answer
    answer_text = example["label"]

    return input_text, answer_text


def parse_answer(response):
    """
    Parse the answer text to get the answer.
    """
    response = response.strip()
    
    # CoT strategy
    if 'boxed{' in response:
        try:
            model_answers = extract_full_boxed_content(response)
            if model_answers:
                # for coding
                # \\boxed{\\text{}}
                try:
                    text_content = re.findall(r'\\text{(.*?)}', model_answers[-1])
                    if text_content:
                        return text_content[-1].strip()
                except Exception:
                    print("Error in extracting text content from boxed answer.")
                return model_answers[-1].strip()
        except Exception:
            print("Error in extracting boxed content.")
            return ""

    # for Coding
    # the correct answer is\n D.
    for flag in ['final answer is', 'correct answer is', 'answer should be', 'answer is', 'answer:']:
        if flag in response.lower():
            try:
                model_answer = response.lower().split(flag)[-1].strip()
                return model_answer.split('\n')[0].split('.')[0]
            except Exception:
                print("Error in extracting answer from response.")
                return ""
    
    return ""


def accuracy(predictions, answers):
    """
    Calculate accuracy of predictions.
    """
    correct = 0
    total = len(predictions)
    
    # parse the predicted answer 
    predictions = [parse_answer(pred).strip() for pred in predictions]

    for prediction, answer in zip(predictions, answers):
        if prediction.lower() == answer.lower():
            correct += 1

    return correct / total if total > 0 else 0.0
    

def folio_scorer(predictions, answers):
    """
    Score the prediction for folio dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict