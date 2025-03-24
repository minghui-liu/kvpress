import pandas as pd
from pathlib import Path
import torch
import json
import os
import logging
from tqdm import tqdm
import re
from typing import Optional

from multiturn.calculate_metrics import calculate_metrics

from transformers import pipeline
from datasets import load_dataset
from fire import Fire

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
    ThinKPress,
    FullPress,
    HashPress,
    PagedAttentionPress,
)


DATASET_DICT = {
    # niah
    "mt_niah_S": "../MT_RULER/save/multi_turn_niah_small.jsonl",
    "mt_niah_M": "../MT_RULER/save/multi_turn_niah_medium.jsonl",
    "mt_niah_L": "../MT_RULER/save/multi_turn_niah_large.jsonl",
    "mt_niah_S_20": "../MT_RULER/save/multi_turn_niah_small_20.jsonl",
    "mt_niah_S_30": "../MT_RULER/save/multi_turn_niah_small_30.jsonl",
    "mt_niah_S_40": "../MT_RULER/save/multi_turn_niah_small_40.jsonl",
    "mt_niah_S_50": "../MT_RULER/save/multi_turn_niah_small_50.jsonl",
    # vt
    "mt_vt_S": "../MT_RULER/save/multi_turn_vt_small.jsonl",
    "mt_vt_M": "../MT_RULER/save/multi_turn_vt_medium.jsonl",
    "mt_vt_L": "../MT_RULER/save/multi_turn_vt_large.jsonl",
    "mt_vt_S_20": "../MT_RULER/save/multi_turn_vt_small_20.jsonl",
    "mt_vt_S_30": "../MT_RULER/save/multi_turn_vt_small_30.jsonl",
    "mt_vt_S_40": "../MT_RULER/save/multi_turn_vt_small_40.jsonl",
    "mt_vt_S_50": "../MT_RULER/save/multi_turn_vt_small_50.jsonl",
    # pr
    "mt_pr_S": "../MT_RULER/save/multi_turn_pr_small.jsonl",
    "mt_pr_M": "../MT_RULER/save/multi_turn_pr_medium.jsonl",
    "mt_pr_L": "../MT_RULER/save/multi_turn_pr_large.jsonl",
    "mt_pr_S_20": "../MT_RULER/save/multi_turn_pr_small_20.jsonl",
    "mt_pr_S_30": "../MT_RULER/save/multi_turn_pr_small_30.jsonl",
    "mt_pr_S_40": "../MT_RULER/save/multi_turn_pr_small_40.jsonl",
    "mt_pr_S_50": "../MT_RULER/save/multi_turn_pr_small_50.jsonl",
}

PRESS_DICT = {
    "expected_attention": ExpectedAttentionPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "tova": TOVAPress(),
    "think": ThinKPress(),
    "full": FullPress(),
    "hash": HashPress(),
    "paged": PagedAttentionPress(),
}

logger = logging.getLogger(__name__)


def evaluate(
    dataset: str,
    data_dir: Optional[str] = None,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device: Optional[str] = None,
    press_name: str = "expected_attention",
    compression_ratio: float = 0.5,
    fraction: float = 1.0,
    max_new_tokens: Optional[int] = None,
    max_context_length: Optional[int] = None,
    save_dir: Optional[str] = None,
):
    """
    Evaluate a model on a dataset using a press and save the results

    Parameters
    ----------
    dataset : str
        Dataset to evaluate
    data_dir : str, optional
        Subdirectory of the dataset to evaluate, by default None
    model : str, optional
        Model to use, by default "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device : str, optional
        Model device, by default cuda:0 if available else cpu. For multi-GPU use "auto"
    press_name : str, optional
        Press to use (see PRESS_DICT), by default "expected_attention"
    compression_ratio : float, optional
        Compression ratio for the press, by default 0.1
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default use the default for the task (recommended)
    fraction : float, optional
        Fraction of the dataset to evaluate, by default 1.0
    max_context_length : int, optional
        Maximum number of tokens to use in the context. By default will use the maximum length supported by the model.
    """

    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    # assert dataset in SCORER_DICT, f"No scorer found for {dataset}"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Save directory
    if save_dir is None:
        save_dir = Path(__file__).parent / "results"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    save_filename = save_dir / (
        "__".join([dataset, model.replace("/", "--"), press_name, str(compression_ratio), 'answers'])
        + ".json"
    )
    score_filename = save_dir / (
        "__".join([dataset, model.replace("/", "--"), press_name, str(compression_ratio), 'score'])
        + ".json"
    )

    if save_filename.exists() and score_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")
        return
    elif save_filename.exists():
        logger.warning(f"LLM answers already exist at {save_filename}")
        # read the answers from save file
        with open(str(save_filename), "r") as f:
            obj = json.load(f)
            predicted_answers = obj["predicted_answers"]
            answers = obj["answers"]
        # Calculate metrics
        metrics = calculate_metrics(predicted_answers, ds["answers"])
        with open(str(score_filename), "w") as f:
            json.dump(metrics, f)
        print(metrics)
        return

    # Load dataframe
    ds = load_dataset("json", data_files=DATASET_DICT[dataset])["train"]
    if fraction < 1.0:
        ds = ds.select(range(int(len(ds) * fraction)))

    # Load press
    assert press_name in PRESS_DICT
    press = PRESS_DICT[press_name]
    press.compression_ratio = compression_ratio

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
            "kv-press-text-generation", model=model, device_map="auto", torch_dtype="auto", model_kwargs=model_kwargs, cache_dir=cache_dir
        )
    else:
        pipe = pipeline(
            "kv-press-text-generation", model=model, device=device, torch_dtype="auto", model_kwargs=model_kwargs, cache_dir=cache_dir
        )

    # Run pipeline on each context
    predicted_answers = []
    for row in tqdm(ds, total=len(ds)):
        context = row["context"]
        questions = row["questions"]
        answer_prefix = row["answer_prefix"] if "answer_prefix" in row else None
        max_new_tokens_ = max_new_tokens if max_new_tokens is not None else 50
        output = pipe(
                context, 
                questions=questions,
                answer_prefix=answer_prefix,
                press=press,
                max_new_tokens=max_new_tokens_,
                max_context_length=max_context_length,
        )
        predicted_answers.append(output["answers"])
        torch.cuda.empty_cache()


    # Save the predicted answer and the ground truth answer to a json file
    save_obj = {
        "predicted_answers": predicted_answers,
        "answers": ds["answers"],
    }
    with open(str(save_filename), "w") as f:
        json.dump(save_obj, f)

    # Calculate metrics
    metrics = calculate_metrics(predicted_answers, ds["answers"])
    with open(str(score_filename), "w") as f:
        json.dump(metrics, f)
    print(metrics)


if __name__ == "__main__":
    cache_dir = "/fs/nexus-scratch/minghui/.cache/huggingface"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    Fire(evaluate)
