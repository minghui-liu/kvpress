
"""
Create a dataset jsonl file for needle in a haystack style value retrieval task.
    with multiple turns of context and questions.
    The dataset contains a block of text that contains variable assignments and value updates.
    The task is to retrieve the value of a variable given a question.
"""
import os
import json
import string
from typing import List, Union
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 

import wonderwords


parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--save_dir", type=Path, default='./output', help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, default='mt_vars_retrieval', help='name of the save dataset jsonl file')
parser.add_argument("--n_samples", type=int, default=100, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
# Complexity Configurations
parser.add_argument("--n_questions", type=int, default=10, help="Number of questions per sample")
parser.add_argument("--n_keys", type=int, default=5, help="Number of keys to ask about in each question")
parser.add_argument("--initial_length", type=int, default=16000, help="Number of words in the initial context")
parser.add_argument("--followup_length", type=int, default=1600, help="Number of words in the followup context")
parser.add_argument("--key_width", type=int, default=5, help="Number of chars in the key")
parser.add_argument("--value_width", type=int, default=5, help="Number of chars in the value")
# parse the arguments
args = parser.parse_args()

random.seed(args.random_seed)

# templates
INITIAL_TEMPLATE = "You will be shown a block of text that contains variable assignments. Make sure to memorize the values of the variables. I will quiz you about them later.\n{content}"
FOLLOWUP_TEMPLATE = "Here is an additional block of text that contains variable assignments. Make sure to memorize the values of the new variables. I will quiz you about them later.\n{content}"
QUESTION_TEMPLATE = "What are the value(s) of variable(s) {key}? Output the values of variable(s) {key}. Do not output anything else."
ANSWER_PREFIX = "The values of {key} is: "
KV_PAIR_TEMPLATE = "var {key} := {value}."

CHARS_PER_KV_PAIR = len(KV_PAIR_TEMPLATE.format(key='a'*args.key_width, value='1'*args.value_width)) + 1 # add 1 for the space seperating the kv pairs
WORDS_PER_KV_PAIR = len(KV_PAIR_TEMPLATE.split())
KV_PAIR_LEN_ESTIMATE = WORDS_PER_KV_PAIR

USED = set()

# # Words
# NOUNS = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
# ADJS = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
# WORDS = [f"{adj}-{noun}" for adj in ADJS for noun in NOUNS]
# WORDS = sorted(list(set(WORDS)))


# def generate_key():
#     # pick a random word from words list
#     key = random.choice(WORDS)
#     while key in USED:
#         key = random.choice(WORDS)
#     USED.add(key)
#     return key   

def generate_key():
    # generate a random args.key_width chars string
    key = ''.join(random.choices(string.ascii_lowercase, k=args.key_width))
    while key in USED:
        key = ''.join(random.choices(string.ascii_lowercase, k=args.key_width))
    USED.add(key)
    return key


def generate_value():
    # generate a random args.value_width chars int
    # val = ''.join(random.choices(string.digits, k=args.value_width))
    start = 10**(args.value_width-1)
    end = (10**args.value_width)-1
    value = str(random.randint(start, end))
    while value in USED:
        value = str(random.randint(start, end))
    USED.add(value)
    return value


def add_kv_pairs(table, n_keys):
    additions = {}
    for i in range(n_keys):
        key = generate_key()
        value = generate_value()
        table[key] = value
        additions[key] = value
    return table, additions


def generate_initial_context():
    table, additions = add_kv_pairs({}, args.initial_length // KV_PAIR_LEN_ESTIMATE)
    kv_pairs = []
    for key, value in additions.items():
        kv_pairs.append(KV_PAIR_TEMPLATE.format(key=key, value=value))
    random.shuffle(kv_pairs)
    return INITIAL_TEMPLATE.format(content=' '.join(kv_pairs)), table


def generate_followup_context(table):
    table, additions = add_kv_pairs(table, args.followup_length // KV_PAIR_LEN_ESTIMATE)
    kv_pairs = []
    for key, value in additions.items():
        kv_pairs.append(KV_PAIR_TEMPLATE.format(key=key, value=value))
    random.shuffle(kv_pairs)
    return FOLLOWUP_TEMPLATE.format(content=' '.join(kv_pairs)), table


def format_multiple(keys):
    if len(keys) == 1:
        return keys[0]
    elif len(keys) == 2:
        return f"{keys[0]} and {keys[1]}"
    else:
        return ', '.join(keys[:-1]) + f", and {keys[-1]}"


def generate_sample():
    USED.clear()
    initial, table = generate_initial_context()
    questions = []
    answers = []
    for _ in range(args.n_questions):
        keys = random.sample(list(table.keys()), args.n_keys)
        followup, table = generate_followup_context(table)
        question = FOLLOWUP_TEMPLATE.format(content=followup) + '\n' + QUESTION_TEMPLATE.format(key=format_multiple(keys))
        answer = [v for key in keys for v in table[key]]
        questions.append(question)
        answers.append(answer)
    return initial, questions, answers


def generate_samples():
    samples = []
    # Generate samples
    for index in tqdm(range(args.n_samples)):
        initial, questions, answers = generate_sample()
        sample = {
            'index': index,
            "context": initial,
            "questions": questions,
            "answers": answers
        }
        samples.append(sample)
    return samples


def write_jsonl(output_path: Union[Path, str], samples: List[dict], ensure_ascii: bool = True):
    with open(output_path, "w", encoding="utf-8") as outfile:
        for s in samples:
            json.dump(s, outfile, ensure_ascii=ensure_ascii)
            outfile.write('\n')


def main():
    save_file = args.save_dir / f'{args.save_name}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)
    samples = generate_samples()
    write_jsonl(save_file, samples)


if __name__ == "__main__":
    main()