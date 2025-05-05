# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from fire import Fire

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import BasePress, KeyRerotationPress, PerLayerCompressionPress

from utils import default_extractor
from gsm8k import gsm8k_formatter, gsm8k_scorer
from folio import folio_formatter, folio_extractor, folio_scorer
from strategyqa import strategyqa_formatter, strategyqa_extractor, strategyqa_scorer
from logiqa import logiqa_formatter, logiqa_scorer
from openbookqa import openbookqa_formatter, openbookqa_extractor, openbookqa_scorer
from aime25 import aime25_formatter, aime25_scorer
from commonsenseqa import commonsenseqa_formatter, commonsenseqa_extractor, commonsenseqa_scorer
from math500 import math500_formatter, math500_scorer

from kvpress import (
    KnormPress,
    RandomPress,
    StreamingLLMPress,
    FullPress,
    H2OPress,
)

logger = logging.getLogger(__name__)

# (dataset_name, subset, split)
DATASET_DICT = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "folio": ("yale-nlp/folio", None, "test"),
    "strategyqa": ("ChilleD/StrategyQA", None, "test"),
    "logiqa": ("lucasmccabe/logiqa", None, "test"),
    "openbookqa": ("allenai/openbookqa", "main", "test"),
    "aime25": ("math-ai/aime25", None, "test"),
    "commonsenseqa": ("tau/commonsense_qa", None, "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
}

FORMATTER_DICT = {
    "gsm8k": gsm8k_formatter,
    "folio": folio_formatter,
    "strategyqa": strategyqa_formatter,
    "logiqa": logiqa_formatter,
    "openbookqa": openbookqa_formatter,
    "aime25": aime25_formatter,
    "commonsenseqa": commonsenseqa_formatter,
    "math500": math500_formatter,
}

EXTRACTOR_DICT = {
    "gsm8k": default_extractor,
    "folio": folio_extractor,
    "strategyqa": strategyqa_extractor,
    "logiqa": default_extractor,
    "openbookqa": openbookqa_extractor,
    "aime25": default_extractor,
    "commonsenseqa": commonsenseqa_extractor,
    "math500": default_extractor,
}

SCORER_DICT = {
    "gsm8k": gsm8k_scorer,
    "folio": folio_scorer,
    "strategyqa": strategyqa_scorer,
    "logiqa": logiqa_scorer,
    "openbookqa": openbookqa_scorer,
    "aime25": aime25_scorer,
    "commonsenseqa": commonsenseqa_scorer,
    "math500": math500_scorer,
}

PRESS_DICT = {
    "knorm": KnormPress(),
    "h2o": H2OPress(),
    "random": RandomPress(),
    "streaming_llm": StreamingLLMPress(),
    "full": FullPress(),
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
    data_split: str = None,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    device: Optional[str] = None,
    press_name: str = "knorm",
    cache_budget: int = 4096,
    fraction: float = 1.0,
    num_samples: int = 0,
    random_seed: int = 42,
    max_new_tokens: Optional[int] = 1024,
    max_context_length: Optional[int] = None,
    do_sampling: bool = True,
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

    hf_name = DATASET_DICT[dataset][0]
    data_dir = DATASET_DICT[dataset][1] if data_dir is None else data_dir
    data_split = DATASET_DICT[dataset][2] if data_split is None else data_split

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)
    save_filename = save_dir / (
        "__".join([dataset, data_dir if data_dir else "", model_name.replace("/", "--"), press_name, f"budget{cache_budget}", f"max_new_tokens{max_new_tokens}"])
        + ".jsonl"
    )
    assert not (fraction < 1.0 and num_samples > 0), "Either fraction or num_samples should be set, not both"
    if num_samples > 0:
        save_filename = save_filename.with_name(save_filename.stem + f"__num_samples{num_samples}" + save_filename.suffix)
    elif fraction < 1.0:
        save_filename = save_filename.with_name(save_filename.stem + f"__fraction{fraction:.2f}" + save_filename.suffix)
    if max_context_length is not None:
        save_filename = save_filename.with_name(save_filename.stem + f"__max_context{max_context_length}" + save_filename.suffix)
    if do_sampling:
        save_filename = save_filename.with_name(save_filename.stem + "__sampling" + save_filename.suffix)
    score_filename = save_dir / (save_filename.stem + "_score.json")

    if save_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")
        print(f"Results already exist. Loading results from {save_filename}")
    else:
        # Load dataset
        ds = load_dataset(hf_name, data_dir=data_dir, split=data_split)
        if num_samples > 0:
            assert num_samples <= len(ds), f"num_samples {num_samples} is larger than the dataset size {len(ds)}"
            ds = ds.shuffle(seed=random_seed).select(range(num_samples))
        elif fraction < 1.0:
            ds = ds.shuffle(seed=random_seed).select(range(int(len(ds) * fraction)))

        # Load press
        assert press_name in PRESS_DICT
        press = PRESS_DICT[press_name]
        formatter = FORMATTER_DICT[dataset]
        extractor = EXTRACTOR_DICT[dataset] 

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
                print("Using flash attention")
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
        save_objs = []
        for i, example in tqdm(enumerate(ds), total=len(ds)):
            input_text, gt_answer_text = formatter(example)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
            if max_context_length is not None:
                inputs = {k: v[:, :max_context_length] for k, v in inputs.items()}
            if max_new_tokens is None:
                max_new_tokens = 16 * 1024 - inputs["input_ids"].shape[1] # use 16k for max length for now

            # Run generation
            if do_sampling:
                with press(model) if press is not None else contextlib.nullcontext():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        use_cache=True,
                        output_attentions=output_attentions(press),
                    )
            else:
                with press(model) if press is not None else contextlib.nullcontext():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        output_attentions=output_attentions(press),
                    )

            pred_start = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][pred_start:], skip_special_tokens=True)
            model_answer = extractor(response)

            # calculate the compression ratio
            input_token_count = inputs["input_ids"].shape[1]
            output_token_count = outputs[0].shape[0] - input_token_count
            total_token_count = outputs[0].shape[0]
            if total_token_count <= cache_budget:
                actual_compression = 1.0
            else:
                actual_compression = cache_budget / total_token_count
            
            save_obj = example.copy()
            save_obj.update(
                {
                    "input_text": input_text,
                    "response": response,
                    "extracted_answer": model_answer,
                    "gt_answer": gt_answer_text,
                    "input_token_count": input_token_count,
                    "output_token_count": output_token_count,
                    "total_token_count": total_token_count,
                    "cache_budget": cache_budget,
                    "compression_ratio": actual_compression,
                }
            )
            save_objs.append(save_obj)

        with open(str(save_filename), "w") as f:
            for obj in save_objs:
                f.write(json.dumps(obj) + "\n")
        print(f"Results saved to {save_filename}")
    # end of the if save_filename.exists()

    # load the results and evaluate the metrics
    with open(str(save_filename), "r") as f:
        save_obj = [json.loads(line) for line in f.readlines()]
    extracted_answers = [obj["extracted_answer"] for obj in save_obj]
    gt_answers = [obj["gt_answer"] for obj in save_obj]

    # Calculate metrics
    scorer = SCORER_DICT[dataset]
    metrics = scorer(extracted_answers, gt_answers)

    # Add average compression ratio
    avg_compression = sum([obj["compression_ratio"] for obj in save_obj]) / len(save_obj)
    metrics["avg_compression"] = avg_compression
    metrics["num_samples"] = len(save_obj)
    metrics["dataset"] = dataset
    metrics["data_split"] = data_split
    metrics["data_dir"] = data_dir
    metrics["model_name"] = model_name
    metrics["press_name"] = press_name
    metrics["cache_budget"] = cache_budget
    metrics["fraction"] = fraction
    metrics["num_samples"] = num_samples
    metrics["max_new_tokens"] = max_new_tokens
    metrics["max_context_length"] = max_context_length
    metrics["random_seed"] = random_seed

    with open(str(score_filename), "w") as f:
        json.dump(metrics, f)
    print(metrics)
    return

if __name__ == "__main__":
    cache_dir = "/fs/nexus-scratch/minghui/.cache/huggingface"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    Fire(evaluate)
