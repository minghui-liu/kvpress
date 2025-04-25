
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

