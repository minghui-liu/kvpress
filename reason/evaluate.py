# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Optional
from time import time

import torch
from datasets import load_dataset
from fire import Fire

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
try:
    from seer_attn import SeerDecodingQwen3ForCausalLM
except ImportError:
    SeerDecodingQwen3ForCausalLM = None
from kvpress import BasePress, KeyRerotationPress, PerLayerCompressionPress

from utils import default_extractor
from gsm8k import gsm8k_formatter, gsm8k_scorer
from keyword_tracker import extract_keywords, tokenize_keywords, track_token_retention
from folio import folio_formatter, folio_extractor, folio_scorer
from strategyqa import strategyqa_formatter, strategyqa_extractor, strategyqa_scorer
from logiqa import logiqa_formatter, logiqa_scorer
from openbookqa import openbookqa_formatter, openbookqa_scorer
from aime25 import aime25_formatter, aime25_scorer
from aime24 import aime24_formatter, aime24_scorer
from commonsenseqa import commonsenseqa_formatter, commonsenseqa_scorer
from math500 import math500_formatter, math500_scorer
from drop import drop_formatter, drop_scorer
from reclor import reclor_formatter, reclor_scorer

from kvpress import (
    KnormPress,
    RandomPress,
    StreamingLLMPress,
    FullPress,
    RKVPress,
    RKVLSHPress,
    H2OPress,
    SnapKVPress,
    NonePress,
)

logger = logging.getLogger(__name__)

# (dataset_name, subset, split)
DATASET_DICT = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "folio": ("yale-nlp/folio", None, "validation"),
    "strategyqa": ("ChilleD/StrategyQA", None, "test"),
    "logiqa": ("lucasmccabe/logiqa", None, "test"),
    "openbookqa": ("allenai/openbookqa", "main", "test"),
    "aime25": ("math-ai/aime25", None, "test"),
    "aime24": ("math-ai/aime24", None, "test"),
    "commonsenseqa": ("tau/commonsense_qa", None, "validation"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "drop": ("ucinlp/drop", None, "validation"),
    "reclor": ("metaeval/reclor", None, "validation"),
}

FORMATTER_DICT = {
    "gsm8k": gsm8k_formatter,
    "folio": folio_formatter,
    "strategyqa": strategyqa_formatter,
    "logiqa": logiqa_formatter,
    "openbookqa": openbookqa_formatter,
    "aime25": aime25_formatter,
    "aime24": aime24_formatter,
    "commonsenseqa": commonsenseqa_formatter,
    "math500": math500_formatter,
    "drop": drop_formatter,
    "reclor": reclor_formatter,
}

EXTRACTOR_DICT = {
    "gsm8k": default_extractor,
    "folio": folio_extractor,
    "strategyqa": strategyqa_extractor,
    "logiqa": default_extractor,
    "openbookqa": default_extractor,
    "aime25": default_extractor,
    "aime24": default_extractor,
    "commonsenseqa": default_extractor,
    "math500": default_extractor,
    "drop": default_extractor,
    "reclor": default_extractor,
}

SCORER_DICT = {
    "gsm8k": gsm8k_scorer,
    "folio": folio_scorer,
    "strategyqa": strategyqa_scorer,
    "logiqa": logiqa_scorer,
    "openbookqa": openbookqa_scorer,
    "aime25": aime25_scorer,
    "aime24": aime24_scorer,
    "commonsenseqa": commonsenseqa_scorer,
    "math500": math500_scorer,
    "drop": drop_scorer,
    "reclor": reclor_scorer,
}

PRESS_DICT = {
    "knorm": KnormPress(),
    "h2o": H2OPress(),
    "random": RandomPress(),
    "streaming_llm": StreamingLLMPress(),
    "snapkv": SnapKVPress(),
    "snapkv_press": SnapKVPress(),  # Alias for snapkv
    "rkv": RKVPress(),
    "rkvlsh": RKVLSHPress(),
    "full": FullPress(),
    "none": NonePress(),  # No-op press that does nothing
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
    cache_budget: int = 512,
    fraction: float = 1.0,
    num_samples: int = 0,
    random_seed: int = 42,
    max_new_tokens: Optional[int] = 2048,
    max_context_length: Optional[int] = None,
    do_sampling: bool = True,
    skip_existing: bool = True,
    compression_ratio: float = 0.1,
    key_channel_compression_ratio: float = 0.5,
    n_hash_buckets: int = 6,
    lam:float=0.1,
    track_tokens: bool = True
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
    cache_budget : int, optional
        Cache budget for the press, by default 512
    compression_ratio : float, optional
        Compression ratio for the press, by default 0.1
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default use the default for the task (recommended)
    fraction : float, optional
        Fraction of the dataset to evaluate, by default 1.0
    num_samples : int, optional
        Number of samples to evaluate, by default 0
    random_seed : int, optional
        Random seed for reproducibility, by default 42
    max_context_length : int, optional
        Maximum number of tokens to use in the context. By default will use the maximum length supported by the model.
    do_sampleing : bool, optional
        Whether to use sampling or not, by default True
    skip_existing : bool, optional
        Whether to skip existing files, by default True
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
    if "rkv" in press_name:
        save_filename = save_dir / (
            "__".join([dataset, data_dir if data_dir else "", model_name.replace("/", "--"), press_name, f"budget{cache_budget}",f"hash_bucket{n_hash_buckets}", f"max_new_tokens{max_new_tokens}",f"lam{int(lam*10)}"])
            + ".jsonl"
        )
    else:
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

    if skip_existing and score_filename.exists():
        logger.warning(f"Score file already exists at {score_filename}, skipping evaluation")
        return

    if skip_existing and save_filename.exists():
        logger.warning(f"Model responses already exist at {save_filename}")
        print(f"Model responses already exist. Loading responses from {save_filename} and evaluating metrics")
    else:
        # Open file for incremental writing (append mode)
        # Clear the file first if it exists
        if save_filename.exists():
            save_filename.unlink()
        # Delete step tracking file if it exists (will be recreated if track_tokens is True)
        if save_filename.with_suffix('.step_tracking.json').exists():
            save_filename.with_suffix('.step_tracking.json').unlink()
        # Load datasetf
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

        # Set the cache budget for the press (NonePress doesn't use it, but set it anyway)
        if press is not None:
            press.cache_budget = cache_budget

        if press_name=="rkvlsh" and press is not None:
            press.n_hash_buckets=n_hash_buckets
            press.lam = lam


        # Initialize pipeline with the correct attention implementation
        model_kwargs = {}
        if press is not None and isinstance(press, H2OPress):
            model_kwargs["attn_implementation"] = "eager"
        else:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using flash attention")
            except ImportError:
                pass

        # Load model
        if "SeerAttention" in model_name:
            # Patch torch.load to handle CPU loading when CUDA is not available
            # This is needed because SeerAttention library loads weights without map_location
            model = SeerDecodingQwen3ForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        seerattn_sparsity_method='token_budget', 
                        seerattn_token_budget = cache_budget 
                    )
            model.to(device)
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model, 
                padding_side="left",
            )
        else:
            # Use torch_dtype instead of dtype for AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
                **model_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Run generation on each context of the dataset
        # Results are written incrementally, so we don't need to store them in memory
        for i, example in tqdm(enumerate(ds), total=len(ds)):
            input_text, gt_answer_text = formatter(example)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
            if max_context_length is not None:
                inputs = {k: v[:, :max_context_length] for k, v in inputs.items()}
            if max_new_tokens is None:
                max_new_tokens = 16 * 1024 - inputs["input_ids"].shape[1] # use 16k for max length for now

            # Memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            start=time()

            # Special handling for SeerAttention with NonePress: use simplified inference path
            # This bypasses all press infrastructure to avoid cache initialization issues
            is_seer_attention_none = (
                "SeerAttention" in model_name
                and (press is None or isinstance(press, NonePress))
            )
            
            # Initialize variables that might be used later
            keywords = {}
            keyword_token_ids = {}
            input_token_ids = []
            
            if is_seer_attention_none:
                # Simplified path: direct generation without any press infrastructure
                # Initialize input_token_ids for potential use in tracking (though tracking is skipped)
                input_token_ids = inputs["input_ids"][0].tolist()
                
                if do_sampling:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        use_cache=True,
                        output_attentions=False,
                    )
                else:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        output_attentions=False,
                    )
            else:
                # Standard path with press infrastructure
                # Reset timing before generation
                if press is not None and not isinstance(press, NonePress):
                    press.reset_timing()
                    # Set tokenizer and input tokens for ranking data collection and per-step tracking
                    if hasattr(press, 'set_tokenizer_and_tokens'):
                        press.set_tokenizer_and_tokens(tokenizer, inputs["input_ids"][0])
                    # Also set tokenizer directly for per-step tracking (for all presses including FullPress)
                    if hasattr(press, 'tokenizer'):
                        press.tokenizer = tokenizer
                    if hasattr(press, 'input_tokens'):
                        press.input_tokens = inputs["input_ids"][0]
                    # For FullPress, ensure tokenizer is set (it doesn't have set_tokenizer_and_tokens)
                    if not hasattr(press, 'tokenizer') or press.tokenizer is None:
                        press.tokenizer = tokenizer
                    if not hasattr(press, 'input_tokens') or press.input_tokens is None:
                        press.input_tokens = inputs["input_ids"][0]
                
                # Extract keywords from input text for tracking
                # Extract keywords only if token tracking is enabled
                if track_tokens:
                    keywords = extract_keywords(input_text)
                    keyword_token_ids = tokenize_keywords(keywords, tokenizer)
                else:
                    keywords = {}
                    keyword_token_ids = {}
                input_token_ids = inputs["input_ids"][0].tolist()

                # Use press context manager
                press_context = press(model) if press is not None and not isinstance(press, NonePress) else contextlib.nullcontext()
                
                if do_sampling:
                    with press_context:
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.7,
                            repetition_penalty=1.2,
                            use_cache=True,
                            output_attentions=output_attentions(press) if press is not None and not isinstance(press, NonePress) else False,
                        )
                else:
                    with press_context:
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                            output_attentions=output_attentions(press) if press is not None and not isinstance(press, NonePress) else False,
                        )

            pred_start = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][pred_start:], skip_special_tokens=True)
            model_answer = extractor(response)

            peak_memory = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            memory_usage=peak_memory / 1024**3
            execution_time=time()-start

            # Get timing metrics from press if available
            timing_metrics = {}
            if press is not None and hasattr(press, 'get_timing_metrics'):
                timing_metrics = press.get_timing_metrics()

            # calculate the compression ratio
            input_token_count = inputs["input_ids"].shape[1]
            output_token_count = outputs[0].shape[0] - input_token_count
            total_token_count = outputs[0].shape[0]
            
            # For NonePress, no compression is applied
            if press is None or isinstance(press, NonePress):
                actual_compression = 1.0
            elif total_token_count <= cache_budget:
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
                    "memory_usage": memory_usage,
                    "execution_time": execution_time,
                }
            )
            
            # Add timing metrics to save_obj
            save_obj.update(timing_metrics)
            
            # Save ranking data if press has ranking collection
            if press is not None and hasattr(press, 'save_all_ranking_data'):
                press.save_all_ranking_data()
            
            # Track keyword retention and token tracking only if enabled
            # Skip all tracking for SeerAttention with NonePress (simplified path)
            if track_tokens and not is_seer_attention_none:
                # Track keyword retention if press tracks retention
                # NonePress doesn't track retention, so skip if it's NonePress
                keyword_retention = {}
                if press is not None and not isinstance(press, NonePress) and hasattr(press, 'get_final_retained_indices'):
                    final_retained_indices = list(press.get_final_retained_indices())
                    if final_retained_indices:
                        retention_results = track_token_retention(
                            input_token_ids,
                            final_retained_indices,
                            keyword_token_ids
                        )
                        keyword_retention = {
                            key_type: {
                                'total_count': results['total_keyword_tokens'],
                                'retained_count': results['retained_keyword_tokens'],
                                'evicted_count': results['evicted_keyword_tokens'],
                                'retention_rate': results['retention_rate']
                            }
                            for key_type, results in retention_results.items()
                        }
                    else:
                        # If no retention tracking, mark all as retained (full cache)
                        keyword_retention = {
                            key_type: {
                                'total_count': len(token_set),
                                'retained_count': len(token_set),
                                'evicted_count': 0,
                                'retention_rate': 1.0
                            }
                            for key_type, token_set in keyword_token_ids.items()
                        }
                else:
                    # For full press or no press, all tokens are retained
                    keyword_retention = {
                        key_type: {
                            'total_count': len(token_set),
                            'retained_count': len(token_set),
                            'evicted_count': 0,
                            'retention_rate': 1.0
                        }
                        for key_type, token_set in keyword_token_ids.items()
                    }
                
                # Add keyword retention to save_obj
                save_obj['keywords'] = keywords
                save_obj['keyword_retention'] = keyword_retention
                
                # Collect per-step token tracking
                # NonePress doesn't track generation steps
                # Skip for SeerAttention with NonePress (simplified path)
                generation_steps = []
                if not is_seer_attention_none and press is not None and not isinstance(press, NonePress) and hasattr(press, 'get_generation_steps'):
                    generation_steps = press.get_generation_steps()
                
                # Save generation_steps to save_obj
                save_obj['generation_steps'] = generation_steps
                
                # Save to a separate detailed JSON file (skip for simplified path)
                if not is_seer_attention_none and generation_steps:
                    step_tracking_file = save_filename.with_suffix('.step_tracking.json')
                    step_data = {
                        'question_index': i,
                        'input_text': input_text,
                        'question_id': example.get('question', '')[:100] if 'question' in example else f'question_{i}',
                        'model_name': model_name,
                        'press_name': press_name,
                        'cache_budget': cache_budget,
                        'generation_steps': generation_steps
                    }
                    # Append to file incrementally (one JSON object per line)
                    with open(str(step_tracking_file), "a", encoding='utf-8') as step_f:
                        step_f.write(json.dumps(step_data, indent=2) + "\n")
            else:
                # Skip token tracking - set empty values
                save_obj['generation_steps'] = []
            
            # Write result incrementally after each example
            with open(str(save_filename), "a", encoding='utf-8') as f:
                f.write(json.dumps(save_obj) + "\n")
            
            print(f"✅ [{i+1}/{len(ds)}] Saved result for question {i+1} to {save_filename.name}")
        
        print(f"\n✅ All results saved to {save_filename}")
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
    
    # Add timing metrics averages
    if save_obj and "prefill_time" in save_obj[0]:
        metrics["avg_prefill_time"] = sum([obj["prefill_time"] for obj in save_obj]) / len(save_obj)
        metrics["avg_decoding_time"] = sum([obj["decoding_time"] for obj in save_obj]) / len(save_obj)
        metrics["avg_total_time"] = sum([obj["total_time"] for obj in save_obj]) / len(save_obj)
        metrics["avg_output_tokens_per_second"] = sum([obj["output_tokens_per_second"] for obj in save_obj]) / len(save_obj)
        metrics["total_prefill_tokens"] = sum([obj["total_prefill_tokens"] for obj in save_obj])
        metrics["total_decoding_tokens"] = sum([obj["total_decoding_tokens"] for obj in save_obj])
    
    metrics["num_samples"] = len(save_obj)
    metrics["dataset"] = dataset
    metrics["data_split"] = data_split
    metrics["data_dir"] = data_dir
    metrics["model_name"] = model_name
    metrics["press_name"] = press_name
    metrics["cache_budget"] = cache_budget
    if press_name=="rkvlsh":
        metrics["n_hash_buckets"] = n_hash_buckets
        metrics["lam"] = lam
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
