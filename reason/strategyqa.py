import re
from utils import extract_full_boxed_content

strategyqa_prompt = "Given some facts and a related question, answer the question with true or false."
# strategyqa_answer_prefix = "Please solve the problem step by step. Answer with true or false and wrap your final answer in one \"\\boxed{{}}\"."
strategyqa_answer_prefix = "Answer with true or false and wrap your final answer in one \"\\boxed{{}}\"."


def strategyqa_formatter(example):
    """
    Format the example for strategyqa dataset.
    """
    input_text = f"{strategyqa_prompt}\Facts:\n{example['facts']}\Question:\n{example['question']}\n{strategyqa_answer_prefix}"
    # parse four # signs and the following text as the answer
    answer_text = str(example["answer"])

    return input_text, answer_text


def parse_answer(response):
    """
    Parse the answer text to get the answer.
    """
    response = response.strip()

    if response.lower() in ['true', 'false']:
        return response.lower()
    
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
    

def strategyqa_scorer(predictions, answers):
    """
    Score the prediction for strategy qa dataset.
    """
    score_dict = {}
    score_dict["accuracy"] = accuracy(predictions, answers)

    return score_dict