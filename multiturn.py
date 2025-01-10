
import torch
from transformers import pipeline
import json
import os

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
)

cache_dir = ".cache/huggingface"
os.environ['HF_HOME'] = cache_dir

# Load pipeline

device = "cuda:0"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# ckpt = "microsoft/Phi-3.5-mini-instruct"
# ckpt = "mistralai/Mistral-Nemo-Instruct-2407"
attn_implementation = "flash_attention_2"  # use "eager" for ObservedAttentionPress and "sdpa" if you can't use "flash_attention_2"
pipe = pipeline("kv-press-text-generation", model=ckpt, device=device, torch_dtype="auto", model_kwargs={"attn_implementation":attn_implementation}, cache_dir=cache_dir)

# Load data
rows = []
with open("dataset/mt_niah.jsonl", "r") as f:
    for line in f:
        rows.append(json.loads(line))
data = rows
# data is a list of dictionaries, each dictionary has the following keys:
# context: a list of strings, each string is a context 
# queries: a list of strings, each string is a query / question
# answers: a list of strings, each string is a ground truth answer
# length: an integer, the length of the context + longest query + longest answer


# Pick a press with a compression ratio, you can run the following cells with different presses
compression_ratio = 0.5
# press = ExpectedAttentionPress(compression_ratio)
# press = KnormPress(compression_ratio)
# press = RandomPress(compression_ratio)
press = SnapKVPress(compression_ratio)

# pick a metric to evaluate the prediction vs ground truth
metric = "rouge"

for i, d in enumerate(data):
    context = d["context"]
    queries = d["queries"]
    answers = d["answers"]
    length = d["length"]

    tokens = pipe.tokenizer.encode(context, return_tensors="pt").to(device)

    pred_answers = pipe(context, questions=queries, press=press)["answers"]
    for question, pred_answer, true_answer in zip(queries, pred_answers, answers):
        print(f"Question:   {question}")
        print(f"Answer:     {true_answer}")
        print(f"Prediction: {pred_answer}")
        print()

    if i > 0:
        break

    # # Evaluate the prediction vs ground truth
    # if metric == "rouge":
    #     from datasets import load_metric
    #     metric = load_metric("rouge")
    #     metric.add_batch(predictions=pred_answers, references=answers)
    #     scores = metric.compute()
    #     print(scores)
    # elif metric == "bleu":
    #     from datasets import load_metric
    #     metric = load_metric("bleu")
    #     metric.add_batch(predictions=pred_answers, references=answers)
    #     scores = metric.compute()
    #     print(scores)
    # elif metric == "bertscore":
    #     from bert_score import score
    #     P, R, F1 = score(pred_answers, answers, lang="en", verbose=True)
    #     print(F1.mean())
    # elif metric == "meteor":
    #     from nltk.translate import meteor_score
    #     scores = [meteor_score.meteor_score([true_answer], pred_answer) for true_answer, pred_answer in zip(answers, pred_answers)]
    #     print(sum(scores) / len(scores))
    # elif metric == "bleurt":
    #     from bleurt import score
    #     scores = score(pred_answers, answers)
    #     print(sum(scores) / len(scores))
    # else:
    #     raise ValueError(f"Metric {metric} not supported")

