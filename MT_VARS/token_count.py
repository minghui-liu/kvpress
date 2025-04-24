import json
import sys
import os

from transformers import AutoTokenizer


# # calculate average tokens of paragraphs
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER = AutoTokenizer.from_pretrained(model_id)
print(f"Successfully loaded tokenizer: {model_id}")


# file = 'save/multi_turn_niah_small.jsonl'
file = sys.argv[1]

if not file.endswith('.jsonl'):
    raise ValueError("Please provide a jsonl file.")
if not os.path.exists(file):
    raise FileNotFoundError(f"File {file} not found.")

# load save/multi_turn_niah_new_test.jsonl
with open(file, 'r') as f:
    data = f.readlines()
    print(f"Loaded {len(data)} samples from {file}.")

data = [json.loads(sample) for sample in data]

context_tokens = []
context_words = []
question_tokens = []
question_words = []
combined_tokens = []
combined_words = []
for i, sample in enumerate(data):
    n_words = len(sample['context'].split())
    n_tokens = len(TOKENIZER.encode(sample['context']))
    total_words = n_words
    total_tokens = n_tokens
    context_tokens.append(n_tokens)
    context_words.append(n_words)
    for q in sample['questions']:
        n_words = len(q.split())
        n_tokens = len(TOKENIZER.encode(q))
        total_tokens += n_tokens
        total_words += n_words
        question_tokens.append(n_tokens)
        question_words.append(n_words)
    combined_tokens.append(total_tokens)
    combined_words.append(total_words)
avg_context_tokens = sum(context_tokens) / len(context_tokens)
avg_context_words = sum(context_words) / len(context_words)
avg_question_tokens = sum(question_tokens) / len(question_tokens)
avg_question_words = sum(question_words) / len(question_words)
avg_combined_tokens = sum(combined_tokens) / len(combined_tokens)
avg_combined_words = sum(combined_words) / len(combined_words)
print(f'Average tokens of context: {avg_context_tokens}, average words of context: {avg_context_words}')
print(f'Averate tokens of questions: {avg_question_tokens}, average words of questions: {avg_question_words}')
print(f'Average combined tokens: {avg_combined_tokens}, average combined words: {avg_combined_words}')