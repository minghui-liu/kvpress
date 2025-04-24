
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
parser.add_argument("--save_name", type=str, default='mt_aggregate', help='name of the save dataset jsonl file')
parser.add_argument("--n_samples", type=int, default=100, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
# Complexity Configurations
parser.add_argument("--n_questions", type=int, default=10, help="Number of questions per sample")
parser.add_argument("--n_keys_per_prefix", type=int, default=5, help="Number of keys in each prefix group")
parser.add_argument("--initial_length", type=int, default=16000, help="Number of words in the initial context")
parser.add_argument("--followup_length", type=int, default=1600, help="Number of words in the followup context")
parser.add_argument("--n_keys_to_add", type=int, default=1, help="Number of keys to add in each followup context")
parser.add_argument("--value_width", type=int, default=3, help="Number of digits in the value")
# parse the arguments
args = parser.parse_args()

random.seed(args.random_seed)

# templates
INITIAL_TEMPLATE = "You will be shown a block of text that contains variable assignments. Some of the variables have a common prefix. Make sure to memorize the name and value of the variables. I will quiz you about them later.\n{content}"
FOLLOWUP_TEMPLATE = "Here is an additional block of text that contains variable assignments. Some of the variables have a common prefix. Make sure to memorize the name and value of the new variables. I will quiz you about them later.\n{content}"
QUESTION_TEMPLATE = "What is the sum of the values of all variables with prefix {prefix}? Output the sum only and do not output anything else."
ANSWER_PREFIX = "The values of {key} is: "
KV_PAIR_TEMPLATE = "var {key} := {value}."

CHARS_PER_KV_PAIR = len(KV_PAIR_TEMPLATE.format(key='abcdef_ghijk', value='1'*args.value_width)) + 1 # add 1 for the space seperating the kv pairs
WORDS_PER_KV_PAIR = len(KV_PAIR_TEMPLATE.split())
KV_PAIR_LEN_ESTIMATE = WORDS_PER_KV_PAIR

USED = set()
USED_PREFIXES = set()

# # Words
NOUNS = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
ADJS = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")


def generate_keys(n, prefix=None):
    # pick a random word from words list
    keys = []
    if prefix is None:
        for _ in range(n):
            adj, noun = random.choice(ADJS), random.choice(NOUNS)
            key = f"{adj}_{noun}"
            while key in USED:
                adj, noun = random.choice(ADJS), random.choice(NOUNS)
                key = f"{adj}_{noun}"
            USED.add(key)
            keys.append(key)
    else:
        for _ in range(n):
            noun = random.choice(NOUNS)
            key = f"{prefix}_{noun}"
            while key in USED:
                noun = random.choice(NOUNS)
                key = f"{prefix}_{noun}"
            USED.add(key)
            keys.append(key)
    return keys


def generate_value():
    # generate a random args.value_width chars int
    # val = ''.join(random.choices(string.digits, k=args.value_width))
    start = 10**(args.value_width-1)
    end = (10**args.value_width)-1
    value = str(random.randint(start, end))
    USED.add(value)
    return value


def add_kv_groups(table, n_groups):
    additions = []
    for i in range(n_groups):
        # generate a random prefix
        prefix = random.choice(ADJS)
        while prefix in USED_PREFIXES:
            prefix = random.choice(ADJS)
        USED_PREFIXES.add(prefix)
        # create a new prefix group
        table[prefix] = []
        # generate args.n_keys_per_prefix keys
        keys = generate_keys(args.n_keys_per_prefix, prefix)
        values = [generate_value() for _ in range(args.n_keys_per_prefix)]
        for key, value in zip(keys, values):
            table[prefix].append((key, value))
            additions.append((key, value))
    return table, additions


def extend_kv_groups(table, n_groups, n_keys):
    additions = []
    # pick n_groups random prefixes
    prefixes = random.sample(list(table.keys()), n_groups)
    for prefix in prefixes:
        # generate n_keys keys
        keys = generate_keys(n_keys, prefix)
        values = [generate_value() for _ in range(n_keys)]
        for key, value in zip(keys, values):
            table[prefix].append((key, value))
            additions.append((key, value))
    return table, additions


def generate_initial_context():
    n_groups = args.initial_length // (args.n_keys_per_prefix * KV_PAIR_LEN_ESTIMATE)
    table, additions = add_kv_groups({}, n_groups)
    strings = [KV_PAIR_TEMPLATE.format(key=key, value=value) for key, value in additions]
    random.shuffle(strings)
    return INITIAL_TEMPLATE.format(content=' '.join(strings)), table


def generate_followup_context(table):
    n_groups_to_extend = args.followup_length // (KV_PAIR_LEN_ESTIMATE * args.n_keys_to_add) 
    table, additions = extend_kv_groups(table, n_groups_to_extend, args.n_keys_to_add)
    strings = [KV_PAIR_TEMPLATE.format(key=key, value=value) for key, value in additions]
    random.shuffle(strings)
    return FOLLOWUP_TEMPLATE.format(content=' '.join(strings)), table


def format_multiple(keys):
    if len(keys) == 1:
        return keys[0]
    elif len(keys) == 2:
        return f"{keys[0]} and {keys[1]}"
    else:
        return ', '.join(keys[:-1]) + f", and {keys[-1]}"


def generate_sample():
    USED.clear()
    USED_PREFIXES.clear()
    initial, table = generate_initial_context()
    questions = []
    answers = []
    for _ in range(args.n_questions):
        prefix = random.choice(list(table.keys()))
        followup, table = generate_followup_context(table)
        question = FOLLOWUP_TEMPLATE.format(content=followup) + '\n' + QUESTION_TEMPLATE.format(prefix=prefix)
        answer = str(sum([int(value) for key, value in table[prefix]]))
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