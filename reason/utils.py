
def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


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
    results = []

    i = 0
    while i < len(s):
        if s[i:i + 7] == r'\boxed{':
            brace_stack = []
            start = i + 7
            i = start

            while i < len(s):
                if s[i] == '{':
                    brace_stack.append(i)
                elif s[i] == '}':
                    if brace_stack:
                        brace_stack.pop()
                    else:
                        results.append(s[start:i])
                        break
                i += 1
        i += 1

    return results


def default_extractor(response):
    """
    Extract the answer from model response.
    """
    response = response.strip()

    # check if the response is a number
    if is_number(response):
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
