import re
from utils import extract_full_boxed_content, is_number

openbookqa_prompt = "Given a piece of fact and a question, pick the correct answer choice."
openbookqa_answer_prefix = "Solve the problem step by step. Answer with the answer choice label and wrap your final answer in \"\\boxed{}\"."


def openbookqa_formatter(example):
    """
    Format the example for openbookqa dataset.
    """
    choice_texts = example['choices']['text']
    choice_labels = example['choices']['label']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])

    input_text = f"{openbookqa_prompt}\nFact:\n{example['fact1']}\nQuestion:\n{example['question_stem']}\nChoices:\n{choices_text}\n{openbookqa_answer_prefix}"
    # parse four # signs and the following text as the answer
    answer_text = example["answerKey"]

    return input_text, answer_text


def openbookqa_extractor(response):
    """
    Parse the answer text to get the answer.
    """
    response = response.strip()

    # check if the response is a number
    if is_number(response):
        return response.strip()
    
    # check if the response is a letter
    if response.isalpha():
        return response.strip()
    
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
            
    for flag in ['option', 'answer', 'option:', 'answer:', 'option.', 'answer.', 'option is', 'correct option is']:
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