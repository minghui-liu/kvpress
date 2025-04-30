# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from kvpress.presses.base_press import BasePress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress
from kvpress.presses.h2o_press import H2OPress
import torch
from datasets import load_dataset
from fire import Fire

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from gsm8k import gsm8k_formatter, gsm8k_scorer
from folio import folio_formatter, folio_scorer
from strategyqa import strategyqa_formatter, strategyqa_scorer
from logiqa import logiqa_formatter, logiqa_scorer
from openbookqa import openbookqa_formatter, openbookqa_scorer
from aime25 import aime25_formatter, aime25_scorer

from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    QFilterPress,
    PyramidKVPress,
    FinchPress,
)

logger = logging.getLogger(__name__)

DATASET_DICT = {
    "gsm8k": "openai/gsm8k",
    "folio": "yale-nlp/folio",
    "strategyqa": "ChilleD/StrategyQA",
    "logiqa": "lucasmccabe/logiqa",
    "openbookqa": "allenai/openbookqa",
    "aime25": "math-ai/aime25",
}

FORMATTER_DICT = {
    "gsm8k": gsm8k_formatter,
    "folio": folio_formatter,
    "strategyqa": strategyqa_formatter,
    "logiqa": logiqa_formatter,
    "openbookqa": openbookqa_formatter,
    "aime25": aime25_formatter,
}

SCORER_DICT = {
    "gsm8k": gsm8k_scorer,
    "folio": folio_scorer,
    "strategyqa": strategyqa_scorer,
    "logiqa": logiqa_scorer,
    "openbookqa": openbookqa_scorer,
    "aime25": aime25_scorer,
}

PRESS_DICT = {
    "knorm": KnormPress(),
    "h2o": H2OPress(),
    "random": RandomPress(),
    "streaming_llm": StreamingLLMPress(),
}


def output_attentions(press: BasePress):
    if isinstance(press, H2OPress):
        return True
    if isinstance(press, (KeyRerotationPress, PerLayerCompressionPress)) and isinstance(
        press.press, H2OPress
    ):
        return True
    return False

def evaluate(
    dataset: str,
    data_dir: Optional[str] = None,
    data_split: str = "test",
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    device: Optional[str] = None,
    press_name: str = "knorm",
    cache_budget: int = 1024,
    fraction: float = 1.0,
    max_new_tokens: Optional[int] = 1024,
    max_context_length: Optional[int] = None,
    compression_ratio: float = 0.1,
    key_channel_compression_ratio: float = 0.5,
):
    """
    Evaluate a model on a dataset using a press and save the results

    Parameters
    ----------
    dataset : str
        Dataset to evaluate
    data_dir : str, optional
        Subdirectory of the dataset to evaluate, by default None
    data_split : str, optional
        Split of the dataset to evaluate, by default "test"
    model_name : str, optional
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
    compress_questions : bool, optional
        Whether to compress the questions as well, by default False
    key_channel_compression_ratio : float, optional
        key Channel Compression ratio for the channel press, by default 0.5
    """

    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    assert dataset in SCORER_DICT, f"No scorer found for {dataset}"
    data_dir = str(data_dir) if data_dir else None

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)
    save_filename = save_dir / (
        "__".join([dataset, data_dir if data_dir else "", model_name.replace("/", "--"), press_name, f"budget{cache_budget}"])
        + ".jsonl"
    )
    if fraction < 1.0:
        save_filename = save_filename.with_name(save_filename.stem + f"__fraction{fraction:.2f}" + save_filename.suffix)
    if max_context_length is not None:
        save_filename = save_filename.with_name(save_filename.stem + f"__max_context{max_context_length}" + save_filename.suffix)
    score_filename = save_dir / (save_filename.stem + "_score.json")

    if save_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")
        print(f"Results already exist. Loading results from {save_filename}")
        # load the results and evaluate the metrics
        with open(str(save_filename), "r") as f:
            save_obj = [json.loads(line) for line in f.readlines()]
        predictions = [obj["response"] for obj in save_obj]
        gt_answers = [obj["gt_answer"] for obj in save_obj]
        metrics = SCORER_DICT[dataset](predictions, gt_answers)
        with open(str(score_filename), "w") as f:
            json.dump(metrics, f)
        print(metrics)
        return

    # Load dataset
    ds = load_dataset(DATASET_DICT[dataset], data_dir=data_dir, split=data_split)
    if fraction < 1.0:
        ds = ds.shuffle(seed=42).select(range(int(len(ds) * fraction)))

    # Load press
    assert press_name in PRESS_DICT
    press = PRESS_DICT[press_name]
    formatter = FORMATTER_DICT[dataset]
    scorer = SCORER_DICT[dataset]

    # Set the cache budget for the press
    press.cache_budget = cache_budget

    # Initialize pipeline with the correct attention implementation
    model_kwargs = {"torch_dtype": "auto"}
    if isinstance(press, H2OPress):
        model_kwargs["attn_implementation"] = "eager"
    else:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Run generation on each context of the dataset
    predictions = []
    gt_answers = []
    save_objs = []
    for i, example in tqdm(enumerate(ds), total=len(ds)):
        input_text, gt_answer_text = formatter(example)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
        if max_context_length is not None:
            inputs = {k: v[:, :max_context_length] for k, v in inputs.items()}
        if max_new_tokens is None:
            max_new_tokens = 16 * 1024 - inputs["input_ids"].shape[1] # use 16k for max length for now

        # Run generation
        with press(model) if press is not None else contextlib.nullcontext():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                output_attentions=output_attentions(press),
            )

        output = outputs[0]
        pred_start = inputs["input_ids"].shape[1]
        pred = tokenizer.decode(output[pred_start:], skip_special_tokens=True)
        predictions.append(pred)
        gt_answers.append(gt_answer_text)
        
        save_obj = example.copy()
        save_obj.update(
            {
                "input_text": input_text,
                "response": pred,
                "gt_answer": gt_answer_text,
            }
        )
        save_objs.append(save_obj)

    with open(str(save_filename), "w") as f:
        for obj in save_objs:
            f.write(json.dumps(obj) + "\n")
    print(f"Results saved to {save_filename}")

    # Calculate metrics
    metrics = scorer(predictions, gt_answers)
    with open(str(score_filename), "w") as f:
        json.dump(metrics, f)
    print(metrics)


if __name__ == "__main__":
    cache_dir = "/fs/nexus-scratch/minghui/.cache/huggingface"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    Fire(evaluate)
