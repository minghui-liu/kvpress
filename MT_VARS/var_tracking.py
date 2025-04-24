
"""
Create a dataset jsonl file for variable tracking.

python multi_turn_vt.py   \
    --save_dir=./save \
    --save_name=multi_turn_vt \
    --n_samples=100 \
    --random_seed=42 \
    --n_questions=10 \
    --initial_length=16000 \
    --followup_length=4000 \
    --initial_hops=5 \
    --n_hops_to_extend=2 \
    --var_width=5 \
    --value_width=5
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
import random
import string
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 


# Basic Configurations
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, default='./output', help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, default='mt_vars_tracking', help='name of the save dataset jsonl file')
parser.add_argument("--n_samples", type=int, default=100, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
# Complexity Configurations
parser.add_argument("--mode", type=str, default='extend', help='mode of the dataset: extend or add')
parser.add_argument("--n_questions", type=int, default=10, help="Number of questions per sample")
parser.add_argument("--initial_length", type=int, default=16000, help="Number of words in the initial context")
parser.add_argument("--followup_length", type=int, default=1600, help="Number of words in the followup context")
parser.add_argument("--initial_hops", type=int, default=2, help="Number of hops in the initial context")
parser.add_argument("--n_hops_to_extend", type=int, default=1, help='number of hops to extend')
parser.add_argument("--var_width", type=int, default=5, help="Number of chars in the variable")
parser.add_argument("--value_width", type=int, default=5, help="Number of chars in the value")
# parse the arguments
args = parser.parse_args()


random.seed(args.random_seed)

# Templates
INITIAL_TEMPLATE = "You will be shown a block of text that contains variable assignments. Each variable can be assigned an integer value or the value of another variable. Memorize and track the values of the variables. I will quiz you about them later.\n{context}"
FOLLOWUP_TEMPLATE = "Here is an additional block of text that contains variable assignments. Each variable can be assigned an integer value or the value of another variable. Memorize and track the values of the variables. I will quiz you about them later.\n{context}"
QUESTION_TEMPLATE = "Find and output all variables that have the value {value}. Do not output anything else."
ANSWER_PREFIX = "According to the variable assignments, the variables that have the value {value} are:\n"
KV_PAIR_TEMPLATE = "var {key} := {value}."

CHARS_PER_KV_PAIR = len(KV_PAIR_TEMPLATE.format(key='a'*args.var_width, value='1'*args.value_width)) + 1 # add 1 for the space seperating the kv pairs
WORDS_PER_KV_PAIR = len(KV_PAIR_TEMPLATE.split())
KV_PAIR_LEN_ESTIMATE = WORDS_PER_KV_PAIR + 1 # add 1 for the VAR prefix

USED = set()

def generate_var():
    # generate a random args.key_width chars string
    var = ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase, k=args.var_width))
    while var in USED:
        var = ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase, k=args.var_width))
    USED.add(var)
    return var


def generate_value():
    # generate a random args.key_width chars int
    # return ''.join(random.choices(string.digits, k=args.value_width))
    start = 10**(args.value_width-1)
    end = (10**args.value_width)-1
    value = str(random.randint(start, end))
    while value in USED:
        value = str(random.randint(start, end))
    USED.add(value)
    return value


def add_chains(table, n_new_chains, n_hops):
    chains = []
    for i in range(n_new_chains):
        value = generate_value()
        vars = [generate_var() for _ in range(n_hops+1)]
        chain = [KV_PAIR_TEMPLATE.format(key=vars[0], value=value)]
        for j in range(n_hops):
            chain.append(KV_PAIR_TEMPLATE.format(key=vars[j+1], value=f'variable {vars[j]}'))
        table[value] = vars
        chains.append(chain)
    return table, chains


def extend_chains(table, n_chains, n_hops):
    chains = []
    # selected = random.sample(list(table.keys()), n_chains) # pick n_chains random chains to extend
    selected = sorted(table, key=lambda x: len(table[x]))[:n_chains] # pick the n_chains shortest chains to extend
    for val in selected:
        vars = table[val]
        chain = []
        for j in range(n_hops): # extend the chain by n_hops
            new_var = generate_var()
            chain.append(KV_PAIR_TEMPLATE.format(key=new_var, value=f'variable {vars[-1]}'))
            vars.append(new_var)
        table[val] = vars
        chains.append(chain)
    return table, chains


def shuffle(chains):
    """
    Randomly shuffle the order of strings within chains
    but keep the order of strings within each chain
    """
    strings = []
    candidates = list(range(len(chains)))
    while candidates:
        i = random.choice(candidates)
        strings.append(chains[i].pop(0))
        if not chains[i]:
            candidates.remove(i)
    return strings


def generate_initial_context():
    # estimate the number of chains needed to fill the initial context
    n_chains = args.initial_length // (KV_PAIR_LEN_ESTIMATE * args.initial_hops + KV_PAIR_LEN_ESTIMATE - 1)
    table, initial_chains = add_chains({}, n_chains, args.initial_hops)
    initial_context = shuffle(initial_chains)
    return initial_context, table
    

def generate_followup_context(table):
    if args.mode == 'add':
        n_chains_to_add = args.followup_length // KV_PAIR_LEN_ESTIMATE // args.initial_hops
        table, followup_chains = add_chains(table, n_chains_to_add, args.initial_hops)
    else:
        n_chains_to_extend = args.followup_length // KV_PAIR_LEN_ESTIMATE // args.n_hops_to_extend
        table, followup_chains = extend_chains(table, n_chains_to_extend, args.n_hops_to_extend)
    followup_context = shuffle(followup_chains)
    return followup_context, table


def generate_sample():
    USED.clear()
    initial_context, table = generate_initial_context()
    questions = []
    answers = []
    for _ in range(args.n_questions):
        followup_context, table = generate_followup_context(table)
        value = random.choice(list(table.keys()))
        question = FOLLOWUP_TEMPLATE.format(context=' '.join(followup_context)) \
            + '\n' + QUESTION_TEMPLATE.format(value=value) # + ' ' + ANSWER_PREFIX.format(value=value)
        answer = table[value].copy()
        questions.append(question)
        answers.append(answer)
    return ' '.join(initial_context), questions, answers


def generate_samples():
    samples = []
    for i in tqdm(range(args.n_samples)):
        initial_context, questions, answers = generate_sample()
        formatted_output = {
            'index': i,
            "context": initial_context,
            "questions": questions,
            "answers": answers,
        }
        samples.append(formatted_output)
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


if __name__=="__main__":
    main()
