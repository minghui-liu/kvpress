import pandas as pd
from pathlib import Path
import torch
import json
import os
import logging
from tqdm import tqdm
import re

from transformers import pipeline
from datasets import load_dataset

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
)


PRESS_DICT = {
    "expected_attention": ExpectedAttentionPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
}


logger = logging.getLogger(__name__)

cache_dir = ".cache/huggingface"
os.environ['HF_HOME'] = cache_dir

device = "cuda:0" if torch.cuda.is_available() else "cpu"


dataset_name = "mt_niah"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "microsoft/Phi-3.5-mini-instruct"
# model_name = "mistralai/Mistral-Nemo-Instruct-2407"
press_name = "expected_attention"

attn_implementation = "flash_attention_2"  # use "eager" for ObservedAttentionPress and "sdpa" if you can't use "flash_attention_2"
# Load pipeline
pipe = pipeline("kv-press-text-generation", model=model_name, device=device, torch_dtype="auto", model_kwargs={"attn_implementation":attn_implementation}, cache_dir=cache_dir)

# Load data
# data is a list of dictionaries, each dictionary has the following keys:
# context: a list of strings, each string is a context 
# queries: a list of strings, each string is a query / question
# answers: a list of strings, each string is a ground truth answer
# length: an integer, the length of the context + longest query + longest answer

# rows = []
# with open("dataset/mt_niah.jsonl", "r") as f:
#     for line in f:
#         rows.append(json.loads(line))
# data = rows


# load dataset
dataset = load_dataset("json", data_files="dataset/mt_niah.jsonl")

# Pick a press with a compression ratio, you can run the following cells with different presses
compression_ratio = 0.1

# Load press
assert press_name in PRESS_DICT
press = PRESS_DICT[press_name]
press.compression_ratio = compression_ratio

# Save directory
save_dir = Path(__file__).parent / "results"
save_dir.mkdir(exist_ok=True)
save_filename = save_dir / (
    "__".join([dataset_name, model_name.replace("/", "--"), press_name, str(compression_ratio)])
    + ".csv"
)
if save_filename.exists():
    logger.warning(f"Results already exist at {save_filename}")

# Initialize pipeline with the correct attention implementation
if isinstance(press, ObservedAttentionPress):
    model_kwargs = {"attn_implementation": "eager"}
else:
    try:
        import flash_attn  # noqa: F401

        model_kwargs = {"attn_implementation": "flash_attention_2"}
    except ImportError:
        model_kwargs = {}

if device == "auto":
    pipe = pipeline(
        "kv-press-text-generation", model=model_name, device_map="auto", torch_dtype="auto", model_kwargs=model_kwargs
    )
else:
    pipe = pipeline(
        "kv-press-text-generation", model=model_name, device=device, torch_dtype="auto", model_kwargs=model_kwargs
    )

df = dataset['train'].to_pandas()


# Run pipeline on each context
df["predicted_answer"] = None
for i, row in tqdm(df.iterrows()):
    context = row["context"]
    queries = list(row["queries"])
    answers = list(row["answers"])
    length = row["length"]

    output = pipe(
            context, 
            questions=queries, 
            press=press
    )
    df.at[i, "predicted_answer"] = output["answers"]
    torch.cuda.empty_cache()

# ----------------- metric functions ----------------- #

def string_match_part(preds, refs):
    score = (
        sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)])
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_all(preds, refs):
    """
    Calculate the string match score for all references
    preds: list of strings, predictions, shape (N,)
    refs: list of list of strings, references, shape (N, M)
    """
    score = (
        sum(
            [sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]
        )
        / len(preds)
        * 100
    )
    return round(score, 2)

def mt_string_match_part(preds, refs):
    """
    Calculate the string match score for all references
    preds: list of list of strings, shape (N, M) where N is the number of samples and M is the number of questions per sample
    refs: list of list of of list of strings, shape (N, M, K) where N is the number of samples, M is the number of questions per sample, and K is the number of references per question
    """
    scores = []
    for pred, ref in zip(preds, refs):
        for p, r in zip(pred, ref):
            score = sum([1.0 if x.lower() in p.lower() else 0.0 for x in r]) / len(r) * 100
            scores.append(score)
    return round(sum(scores) / len(scores), 2)


# def calculate_metrics(preds, refs, metric_fn):
#     score = metric_fn(preds, refs)
#     return score


def calculate_metrics(df: pd.DataFrame) -> dict:
    scores = {}
    np_pattern = re.compile(r"[\x00-\x1f]")
    df["predicted_answer"] = df["predicted_answer"].apply(lambda lst: [np_pattern.sub("", x.strip()).strip() for x in lst])

    metric_fn = mt_string_match_part
    preds = df["predicted_answer"].tolist()
    refs = df["answers"].tolist()
    score = metric_fn(preds, refs)
    scores["all"] = {"string_match": score}
    return scores



# # Calculate metrics
# metric_fn = string_match_all

# score = calculate_metrics(all_pred, all_gt, metric_fn)

# print(f"String match score: {score}")



# Calculate metrics
# scorer = SCORER_DICT[dataset]
# from evaluation.ruler.calculate_metrics import calculate_metrics as ruler_scorer
# scorer = ruler_scorer

scorer = calculate_metrics
metrics = scorer(df)
with open(str(save_filename).replace(".csv", ".json"), "w") as f:
    json.dump(metrics, f)
print(metrics)

