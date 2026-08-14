
from answer_grading import extract_boxed_content, extract_final_answer


def verify_extraction(extraction):
    return extraction is not None and str(extraction).strip() != ""


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def extract_full_boxed_content(s):
    """
    Extract the full content inside \boxed{}, handling nested braces {{}} properly.
    """
    return extract_boxed_content(s)


def default_extractor(response):
    """
    Extract the answer from model response.
    """
    return extract_final_answer(response)
