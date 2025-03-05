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
    FullPress,
    HashPress
)

DATASET_DICT = {
    "mt_niah": "../MT_RULER/save/multi_turn_niah/validation.jsonl",
    "mt_vt": "../MT_RULER/save/multi_turn_vt/validation.jsonl",
    "mt_passage": "../MT_RULER/save/multi_turn_passage/validation.jsonl",
    "niah_simple": "../MT_RULER/save/niah_simple/niah_simple.jsonl",
}

PRESS_DICT = {
    "expected_attention": ExpectedAttentionPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "full": FullPress(),
    "hash": HashPress(),
}

SCORER_DICT = {
    "mt_niah": calculate_metrics,
    "mt_vt": calculate_metrics,
    "mt_passage": calculate_metrics,
    "niah_simple": calculate_metrics,
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
    data_dir = str(data_dir) if data_dir else None

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Save directory
    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)
    save_filename = save_dir / (
        "__".join([dataset, data_dir if data_dir else "", model.replace("/", "--"), press_name, str(compression_ratio)])
        + ".csv"
    )
    if save_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")

    # Load dataframe
    # df = load_dataset("json", data_files=DATASET_DICT[dataset], data_dir=data_dir).to_pandas()
    df = load_dataset("json", data_files=DATASET_DICT[dataset])["train"].to_pandas()
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)

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
    df["predicted_answer"] = None
    df["task"] = dataset
    for i, row in tqdm(df.iterrows()):
        context = row["context"]
        questions = list(row["questions"])
        # answers = list(row["answers"])
        # length = row["length"] if "length" in row else None
        answer_prefix = row["answer_prefix"] if "answer_prefix" in row else None
        # TODO: hack right now, please fix in the dataset generation stage
        if dataset == "mt_passage":
            answer_prefix = "Please enter the number of the paragraph that the abstract is from. The answer format must be like “Paragraph 1”, “Paragraph 2”, etc.\nThe answer is:"
        # max_new_tokens_ = max_new_tokens if max_new_tokens is not None else df_["max_new_tokens"].iloc[0]
        output = pipe(
                context, 
                questions=questions, 
                answer_prefix=answer_prefix,
                press=press,
                # max_new_tokens=max_new_tokens_,
                max_context_length=max_context_length,
        )
        df.at[i, "predicted_answer"] = output["answers"]
        torch.cuda.empty_cache()

    # Save answers
    df["predicted_answer"].to_csv(str(save_filename), index=False)
    print(f"Saved answers to {save_filename}")

    # Calculate metrics
    scorer = SCORER_DICT[dataset]
    metrics = scorer(df)
    with open(str(save_filename).replace(".csv", ".json"), "w") as f:
        json.dump(metrics, f)
    print(metrics)


if __name__ == "__main__":
    cache_dir = "../.cache/huggingface"
    os.environ['HF_HOME'] = cache_dir
    Fire(evaluate)
