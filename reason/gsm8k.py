import re
from utils import extract_full_boxed_content, is_number

gsm8k_prompt = "\nSolve the problem step by step. Wrap your final answer in \"\\boxed{}\"."


def gsm8k_formatter(example):
    """
    Format the example for GSM8K dataset.
    """
    question_text = example["question"] + gsm8k_prompt
    # parse four # signs and the following text as the answer
    answer_text = example["answer"].split("####")[-1].strip()

    return question_text, answer_text


def parse_answer(response):
    """
    Parse the answer text to get the answer.
    """
    response = response.strip()

    # Direct Strategy Open-ended
    # 1
    if is_number(response):
        return response
    
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
        if prediction == answer:
            correct += 1

    return correct / total if total > 0 else 0.0
    

def gsm8k_scorer(predictions, answers):
    """
    Score the prediction for GSM8K dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict