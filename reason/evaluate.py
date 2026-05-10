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
    from seer_attn import SeerDecodingQwen2ForCausalLM
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
from gpqa import gpqa_formatter, gpqa_extractor, gpqa_scorer
from gpqa_diamond import (
    gpqa_diamond_formatter,
    gpqa_diamond_extractor,
    gpqa_diamond_scorer,
)

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
    TurboQuantPress,
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
    "gpqa": ("Idavidrein/gpqa", "gpqa_main", "train"),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
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
    "gpqa": gpqa_formatter,
    "gpqa_diamond": gpqa_diamond_formatter,
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
    "gpqa": gpqa_extractor,
    "gpqa_diamond": gpqa_diamond_extractor,
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
    "gpqa": gpqa_scorer,
    "gpqa_diamond": gpqa_diamond_scorer,
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
    "turboquant": TurboQuantPress(),
}


def output_attentions(press: BasePress):
    if isinstance(press, (H2OPress, KnormPress, StreamingLLMPress)):
        return True
    if isinstance(press, (KeyRerotationPress, PerLayerCompressionPress)) and isinstance(
        press.press, (H2OPress, KnormPress, StreamingLLMPress)
    ):
        return True
    return False


def load_reason_dataset(hf_name: str, subset: Optional[str], split: str):
    """
    Load a dataset split with robust subset handling.

    `subset` in this codebase is typically a Hugging Face config name (e.g. "main",
    "gpqa_diamond"). Some legacy datasets may instead expect `data_dir`, so we
    progressively retry with sensible fallbacks.
    """
    if subset is None:
        return load_dataset(hf_name, split=split)

    load_errors = []

    # 1) Preferred: treat subset as HF config name
    try:
        return load_dataset(hf_name, name=subset, split=split)
    except Exception as exc:
        load_errors.append(("name", exc))

    # 2) Fallback: treat subset as data directory
    try:
        return load_dataset(hf_name, data_dir=subset, split=split)
    except Exception as exc:
        load_errors.append(("data_dir", exc))

    # 3) Only if no subset was requested would we use default config.
    # Since subset is explicitly provided here, fail loudly to avoid
    # evaluating the wrong dataset/config by accident.

    attempts = ", ".join(mode for mode, _ in load_errors)
    errors = " | ".join(f"{mode}:{type(err).__name__}: {err}" for mode, err in load_errors)
    raise RuntimeError(
        f"Failed to load dataset '{hf_name}' with subset '{subset}' and split '{split}'. "
        f"Tried [{attempts}]. Details: {errors}. "
        "If this dataset is gated, accept its terms on Hugging Face and ensure you are logged in."
    ) from load_errors[-1][1]

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
    dataset_block_index: int = 0,
    dataset_block_size: int = 50,
    random_seed: int = 42,
    max_new_tokens: Optional[int] = 2048,
    max_context_length: Optional[int] = None,
    do_sampling: bool = True,
    skip_existing: bool = True,
    compression_ratio: float = 0.1,
    key_channel_compression_ratio: float = 0.5,
    n_hash_buckets: int = 6,
    lam:float=0.1,
    n_bits: int = 4,
    track_tokens: bool = False,
    track_buckets: bool = False,
    enable_qualitative_analysis: bool = False,
    measure_memory: bool = True,
    measure_latency: bool = True,
    temperature: float = 0.6,
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
    dataset_block_index : int, optional
        1-based block index to evaluate within the selected dataset subset. If 0, evaluate all selected samples.
    dataset_block_size : int, optional
        Number of samples per block when dataset_block_index is set, by default 50
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
    enable_qualitative_analysis : bool, optional
        Enable qualitative token retention/eviction analysis for RKV-LSH, by default False
    measure_memory : bool, optional
        Whether to measure GPU memory usage, by default True
    measure_latency : bool, optional
        Whether to measure execution latency, by default True
    """

    # Fire may pass boolean args as strings ("true"/"false") — normalize them
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    track_tokens = _to_bool(track_tokens)
    track_buckets = _to_bool(track_buckets)
    enable_qualitative_analysis = _to_bool(enable_qualitative_analysis)
    measure_memory = _to_bool(measure_memory)
    measure_latency = _to_bool(measure_latency)
    do_sampling = _to_bool(do_sampling)
    skip_existing = _to_bool(skip_existing)

    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    assert dataset in SCORER_DICT, f"No scorer found for {dataset}"
    assert dataset_block_index >= 0, "dataset_block_index must be >= 0"
    assert dataset_block_size > 0, "dataset_block_size must be > 0"

    hf_name = DATASET_DICT[dataset][0]
    data_dir = DATASET_DICT[dataset][1] if data_dir is None else data_dir
    data_split = DATASET_DICT[dataset][2] if data_split is None else data_split

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)
    if "rkv" in press_name:
        # Format lambda with 3 decimal places, then format for filename
        # e.g., 0.01 -> "001", 0.05 -> "005", 0.1 -> "01", 1.0 -> "1"
        # Multiply by 100 to get integer representation, then format with leading zeros if needed
        lam_int = int(round(lam * 100))
        if lam_int == 0:
            lam_sanitized = "0"
        elif lam_int < 10:
            lam_sanitized = f"00{lam_int}"  # 0.01 -> 001, 0.05 -> 005
        elif lam_int < 100:
            lam_sanitized = f"0{lam_int}"    # 0.1 -> 01, 0.2 -> 02
        else:
            lam_sanitized = str(lam_int)     # 1.0 -> 100, but we want "1"
            # Remove trailing zeros: 100 -> 1, 200 -> 2
            lam_sanitized = str(int(lam_sanitized) // 100) if lam_int % 100 == 0 else lam_sanitized
        save_filename = save_dir / (
            "__".join([dataset, data_dir if data_dir else "", model_name.replace("/", "--"), press_name, f"budget{cache_budget}",f"hash_bucket{n_hash_buckets}", f"max_new_tokens{max_new_tokens}",f"lam{lam_sanitized}"])
            + ".jsonl"
        )
    elif press_name == "turboquant":
        save_filename = save_dir / (
            "__".join([dataset, data_dir if data_dir else "", model_name.replace("/", "--"), press_name, f"int{n_bits}", f"max_new_tokens{max_new_tokens}"])
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
    if dataset_block_index > 0:
        save_filename = save_filename.with_name(
            save_filename.stem + f"__block{dataset_block_index}_size{dataset_block_size}" + save_filename.suffix
        )
    if num_samples > 0 or fraction < 1.0:
        save_filename = save_filename.with_name(save_filename.stem + f"__seed{random_seed}" + save_filename.suffix)
    if max_context_length is not None:
        save_filename = save_filename.with_name(save_filename.stem + f"__max_context{max_context_length}" + save_filename.suffix)
    if do_sampling:
        save_filename = save_filename.with_name(save_filename.stem + "__sampling" + save_filename.suffix)
    score_filename = save_dir / (save_filename.stem + "_score.json")
    evaluated_sample_start = 1
    evaluated_sample_end = 0

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
        # Load datasetf
        ds = load_reason_dataset(hf_name, data_dir, data_split)
        if num_samples > 0:
            assert num_samples <= len(ds), f"num_samples {num_samples} is larger than the dataset size {len(ds)}"
            ds = ds.shuffle(seed=random_seed).select(range(num_samples))
        elif fraction < 1.0:
            ds = ds.shuffle(seed=random_seed).select(range(int(len(ds) * fraction)))

        selected_dataset_size = len(ds)
        evaluated_sample_start = 1
        evaluated_sample_end = selected_dataset_size
        if dataset_block_index > 0:
            block_start = (dataset_block_index - 1) * dataset_block_size
            if block_start >= selected_dataset_size:
                raise ValueError(
                    f"dataset_block_index {dataset_block_index} with dataset_block_size {dataset_block_size} "
                    f"starts at sample {block_start + 1}, which exceeds the selected dataset size {selected_dataset_size}."
                )
            block_end = min(block_start + dataset_block_size, selected_dataset_size)
            ds = ds.select(range(block_start, block_end))
            evaluated_sample_start = block_start + 1
            evaluated_sample_end = block_end

        # Load press
        assert press_name in PRESS_DICT
        press = PRESS_DICT[press_name]
        formatter = FORMATTER_DICT[dataset]
        extractor = EXTRACTOR_DICT[dataset] 

        # Set the cache budget for the press (NonePress doesn't use it, but set it anyway)
        if press is not None:
            press.cache_budget = cache_budget
            # Set measure_latency flag to control internal timing
            if hasattr(press, 'measure_latency'):
                press.measure_latency = measure_latency

        if press_name == "turboquant" and press is not None:
            press.n_bits = n_bits
            press.cache_budget = 0  # use n_bits directly, not budget-derived bits

        if press_name=="rkvlsh" and press is not None:
            press.n_hash_buckets=n_hash_buckets
            press.lam = lam
            press.initialize_buckets(device=device)
            # Enable bucket tracking if requested
            if track_buckets:
                press.enable_bucket_tracking()

        # Presses that consume attentions need eager attention.
        attention_config = {}
        if output_attentions(press):
            attention_config = {
                "output_attentions": True,
                "attn_implementation": "eager",
            }
        
        if "SeerAttention" in model_name:
            # SeerAttention models: Load config first, then tokenizer from base_model
            # This is the recommended approach per SeerAttention documentation
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model,
                trust_remote_code=True,
                padding_side="left",
            )
            model = SeerDecodingQwen2ForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                seerattn_sparsity_method='token_budget',
                seerattn_token_budget=cache_budget,
                **attention_config
            )
            model.to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                **attention_config
            )
        
        # Set pad token to eos token if not already set (required for generation)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special token artifact characters so skip_special_tokens will filter them
        # These appear in tokenizer outputs even though skip_special_tokens=True is used
        special_artifacts = ["\u0120", "\u010a", "\u0109"]
        special_tokens_to_add = []
        for artifact in special_artifacts:
            if artifact not in tokenizer.all_special_tokens:
                special_tokens_to_add.append(artifact)
        if special_tokens_to_add:
            # Add these as additional special tokens
            tokenizer.add_special_tokens({
                "additional_special_tokens": special_tokens_to_add
            })


        # Run generation on each context of the dataset
        # Results are written incrementally, so we don't need to store them in memory
        for i, example in tqdm(enumerate(ds), total=len(ds)):
            # Aggressive memory cleanup at the START of each sample to prevent accumulation
            import gc
            torch.cuda.empty_cache()
            if measure_memory:  # Only synchronize if measuring memory
                torch.cuda.synchronize()
            gc.collect()
            
            input_text, gt_answer_text = formatter(example)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
            if max_context_length is not None:
                inputs = {k: v[:, :max_context_length] for k, v in inputs.items()}
            if max_new_tokens is None:
                max_new_tokens = 16 * 1024 - inputs["input_ids"].shape[1] # use 16k for max length for now

            # Reset memory stats and clear cache before generation
            if measure_memory:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            # Synchronize before timing only if measuring latency for accurate measurements
            if measure_latency:
                torch.cuda.synchronize()
                if not measure_memory:  # Only clear cache if not already done for memory measurement
                    torch.cuda.empty_cache()
                start=time()
            else:
                start = None

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
                        temperature=temperature,
                        repetition_penalty=1.2,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        output_attentions=False,
                    )
                else:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        output_attentions=False,
                    )
                
                # Decode response for SeerAttention path
                pred_start = inputs["input_ids"].shape[1]
                response = tokenizer.decode(outputs[0][pred_start:], skip_special_tokens=True)
                # Clean up special token artifacts (\u0120 = Ġ, \u010a = Ċ, etc.)
                response = response.replace("\u0120", " ").replace("\u010a", " ").replace("\u0109", " ")
                # Remove other control characters while preserving spaces and newlines
                response = "".join(c if ord(c) >= 32 or c in "\n\r\t" else "" for c in response)
                model_answer = extractor(response)
            else:
                # Standard path with press infrastructure
                # Reset timing and clear any accumulated state before generation
                if press is not None and not isinstance(press, NonePress):
                    press.reset_timing()
                    # Clear any accumulated hidden states or cached tensors
                    if hasattr(press, 'acc_hidden_states') and press.acc_hidden_states is not None:
                        del press.acc_hidden_states
                        press.acc_hidden_states = None
                    if hasattr(press, 'accumulated_tokens'):
                        press.accumulated_tokens = 0
                    # Clear cached tensors in press if they exist
                    if hasattr(press, 'cos_bucket_cached'):
                        press.cos_bucket_cached = None
                    if hasattr(press, 'powers_of_two_cached'):
                        press.powers_of_two_cached = None
                    # Reset bucket counts for new sample if tracking
                    if track_buckets and hasattr(press, 'reset_bucket_counts'):
                        press.reset_bucket_counts()
                    
                    # Always set tokenizer and input tokens for tracking
                    if track_tokens or enable_qualitative_analysis:
                        press.tokenizer = tokenizer
                        press.input_tokens = inputs["input_ids"][0]
                        if hasattr(press, 'set_tokenizer_and_tokens'):
                            press.set_tokenizer_and_tokens(tokenizer, inputs["input_ids"][0])
                    else:
                        press.tokenizer = None
                        press.input_tokens = None
            
            # Extract keywords from input text for tracking
            # Extract keywords ONLY if track_tokens is explicitly True
            if track_tokens:
                keywords = extract_keywords(input_text)
                keyword_token_ids = tokenize_keywords(keywords, tokenizer)
            else:
                keywords = {}
                keyword_token_ids = {}
            input_token_ids = inputs["input_ids"][0].tolist()

            # Only run standard path generation if not using SeerAttention simplified path
            if not is_seer_attention_none:
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
                            temperature=temperature,
                            repetition_penalty=1.2,
                            use_cache=True,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
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
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            output_attentions=output_attentions(press) if press is not None and not isinstance(press, NonePress) else False,
                        )
                
                # Decode response for standard path
                pred_start = inputs["input_ids"].shape[1]
                response = tokenizer.decode(outputs[0][pred_start:], skip_special_tokens=True)
                # Clean up special token artifacts (\u0120 = Ġ, \u010a = Ċ, etc.)
                response = response.replace("\u0120", " ").replace("\u010a", " ").replace("\u0109", " ")
                # Remove other control characters while preserving spaces and newlines
                response = "".join(c if ord(c) >= 32 or c in "\n\r\t" else "" for c in response)
                model_answer = extractor(response)

            # Get timing metrics from press if available (before deleting tensors)
            timing_metrics = {}
            if press is not None and hasattr(press, 'get_timing_metrics'):
                timing_metrics = press.get_timing_metrics()

            # Calculate metrics before deleting tensors
            input_token_count = inputs["input_ids"].shape[1]
            output_token_count = outputs[0].shape[0] - input_token_count
            response_token_length = output_token_count
            total_token_count = outputs[0].shape[0]
            
            # Measure memory only if requested
            if measure_memory:
                if measure_latency:
                    torch.cuda.synchronize()  # Ensure all operations complete before reading memory
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage = peak_memory / 1024**3
            else:
                memory_usage = 0.0
            
            # Measure latency only if requested
            if measure_latency:
                torch.cuda.synchronize()  # Ensure all operations complete before timing
                execution_time = time() - start
            else:
                execution_time = 0.0
            
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
                    "response_token_length": response_token_length,
                    "total_token_count": total_token_count,
                    "cache_budget": cache_budget,
                    "compression_ratio": actual_compression,
                    "memory_usage": memory_usage,
                    "execution_time": execution_time,
                }
            )
            
            # Aggressive memory cleanup after each sample
            # Delete large tensors explicitly (after all metrics and save_obj are calculated)
            del outputs
            del inputs
            del response
            if 'model_answer' in locals():
                del model_answer
            
            # Clear press accumulated state after each sample
            if press is not None and not isinstance(press, NonePress):
                if hasattr(press, 'acc_hidden_states') and press.acc_hidden_states is not None:
                    del press.acc_hidden_states
                    press.acc_hidden_states = None
                if hasattr(press, 'accumulated_tokens'):
                    press.accumulated_tokens = 0
                # Clear any cached ranking data to prevent accumulation
                if hasattr(press, 'ranking_data'):
                    press.ranking_data = []
            
            # Clear CUDA cache and reset stats
            torch.cuda.empty_cache()
            if measure_memory:  # Only synchronize and reset if measuring memory
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Force Python garbage collection (multiple passes for thorough cleanup)
            import gc
            gc.collect()
            gc.collect()  # Second pass to catch circular references
            
            # Add timing metrics to save_obj
            save_obj.update(timing_metrics)
            
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
                
                generation_steps = []
                if not is_seer_attention_none and press is not None and not isinstance(press, NonePress) and hasattr(press, 'get_generation_steps'):
                    generation_steps = press.get_generation_steps()
                
                # Save generation_steps to save_obj
                save_obj['generation_steps'] = generation_steps
            else:
                save_obj['keywords'] = {}
                save_obj['keyword_retention'] = {}
                save_obj['generation_steps'] = []
            
            # Add bucket counts if tracking is enabled
            if track_buckets and press is not None and hasattr(press, 'get_bucket_counts'):
                bucket_counts = press.get_bucket_counts()
                if bucket_counts is not None:
                    save_obj['bucket_counts'] = bucket_counts.tolist()
                    save_obj['sample_id'] = i
            
            # Write result incrementally after each example
            with open(str(save_filename), "a", encoding='utf-8') as f:
                f.write(json.dumps(save_obj) + "\n")
            
            # Clear save_obj to free memory
            del save_obj

            # Advance qualitative sample state without writing auxiliary files.
            if enable_qualitative_analysis and press_name in ["rkv", "rkvlsh"] and press is not None:
                if hasattr(press, 'next_sample'):
                    press.next_sample()

            print(f"✅ [{i+1}/{len(ds)}] Saved result for question {i+1} to {save_filename.name} (Memory: {memory_usage:.2f} GB)")
            
            # Additional aggressive memory cleanup every 3 samples to prevent accumulation
            if (i + 1) % 3 == 0:
                # Clear all press state
                if press is not None and not isinstance(press, NonePress):
                    if hasattr(press, 'acc_hidden_states') and press.acc_hidden_states is not None:
                        del press.acc_hidden_states
                        press.acc_hidden_states = None
                    if hasattr(press, 'ranking_data'):
                        press.ranking_data = []
                    # Clear cached tensors
                    if hasattr(press, 'cos_bucket_cached'):
                        press.cos_bucket_cached = None
                    if hasattr(press, 'powers_of_two_cached'):
                        press.powers_of_two_cached = None
                    if hasattr(press, 'proj_matrix'):
                        # Keep proj_matrix but clear device-specific caches
                        pass
                
                # Aggressive CUDA cleanup
                torch.cuda.empty_cache()
                if measure_memory:  # Only synchronize and reset if measuring memory
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                import gc
                gc.collect()
                gc.collect()  # Second pass
                print(f"   🧹 Aggressive memory cleanup after {i+1} samples")

        print(f"\n✅ All results saved to {save_filename}")
    # end of the if save_filename.exists()

    # load the results and evaluate the metrics
    with open(str(save_filename), "r") as f:
        save_obj = [json.loads(line) for line in f.readlines()]
    extracted_answers = [obj["extracted_answer"] for obj in save_obj]
    gt_answers = [obj["gt_answer"] for obj in save_obj]

    if save_obj:
        if dataset_block_index > 0:
            evaluated_sample_start = (dataset_block_index - 1) * dataset_block_size + 1
            evaluated_sample_end = evaluated_sample_start + len(save_obj) - 1
        else:
            evaluated_sample_start = 1
            evaluated_sample_end = len(save_obj)

    # Calculate metrics
    scorer = SCORER_DICT[dataset]
    metrics = scorer(extracted_answers, gt_answers)

    # Add average compression ratio
    avg_compression = sum([obj["compression_ratio"] for obj in save_obj]) / len(save_obj)
    metrics["avg_compression"] = avg_compression

    # Add actual response token counts (generated answer length, excluding model-internal decoding accounting)
    metrics["total_output_tokens_generated"] = sum([obj["output_token_count"] for obj in save_obj])
    metrics["avg_output_tokens_generated_per_sample"] = metrics["total_output_tokens_generated"] / len(save_obj) if len(save_obj) > 0 else 0
    response_token_lengths = [obj.get("response_token_length", obj["output_token_count"]) for obj in save_obj]
    metrics["total_response_token_length"] = sum(response_token_lengths)
    metrics["avg_response_token_length"] = metrics["total_response_token_length"] / len(save_obj) if len(save_obj) > 0 else 0
    metrics["min_response_token_length"] = min(response_token_lengths) if response_token_lengths else 0
    metrics["max_response_token_length"] = max(response_token_lengths) if response_token_lengths else 0
    if response_token_lengths:
        import statistics
        metrics["std_response_token_length"] = statistics.stdev(response_token_lengths) if len(response_token_lengths) > 1 else 0.0
    else:
        metrics["std_response_token_length"] = 0.0
    metrics["num_responses_reaching_max_output_tokens"] = (
        sum(1 for length in response_token_lengths if max_new_tokens is not None and length >= max_new_tokens)
        if response_token_lengths
        else 0
    )
    
    # Add memory metrics if measured
    if measure_memory and save_obj:
        metrics["avg_memory_usage_gb"] = sum([obj.get("memory_usage", 0.0) for obj in save_obj]) / len(save_obj)
        metrics["max_memory_usage_gb"] = max([obj.get("memory_usage", 0.0) for obj in save_obj])
    
    # Add latency metrics if measured
    if measure_latency and save_obj:
        metrics["avg_execution_time"] = sum([obj.get("execution_time", 0.0) for obj in save_obj]) / len(save_obj)
        metrics["total_execution_time"] = sum([obj.get("execution_time", 0.0) for obj in save_obj])
    
    # Add timing metrics averages (from press internal timing)
    if save_obj and "prefill_time" in save_obj[0]:
        import numpy as np
        
        prefill_times = [obj["prefill_time"] for obj in save_obj]
        decoding_times = [obj["decoding_time"] for obj in save_obj]
        total_times = [obj["total_time"] for obj in save_obj]
        throughputs = [obj["output_tokens_per_second"] for obj in save_obj]
        
        metrics["avg_prefill_time"] = sum(prefill_times) / len(prefill_times)
        metrics["avg_decoding_time"] = sum(decoding_times) / len(decoding_times)
        metrics["avg_total_time"] = sum(total_times) / len(total_times)
        metrics["avg_output_tokens_per_second"] = sum(throughputs) / len(throughputs)
        
        # Add percentile metrics for latency
        metrics["p90_decoding_time"] = float(np.percentile(decoding_times, 90))
        metrics["p99_decoding_time"] = float(np.percentile(decoding_times, 99))
        metrics["p90_total_time"] = float(np.percentile(total_times, 90))
        metrics["p99_total_time"] = float(np.percentile(total_times, 99))
        metrics["p90_throughput"] = float(np.percentile(throughputs, 10))  # Lower is worse for throughput
        metrics["p99_throughput"] = float(np.percentile(throughputs, 1))   # Lower is worse for throughput
        
        metrics["total_prefill_tokens"] = sum([obj["total_prefill_tokens"] for obj in save_obj])
        metrics["total_decoding_tokens"] = sum([obj["total_decoding_tokens"] for obj in save_obj])

        # Add average token counts per sample
        metrics["avg_prefill_tokens_per_sample"] = metrics["total_prefill_tokens"] / len(save_obj) if len(save_obj) > 0 else 0
        metrics["avg_decoding_tokens_per_sample"] = metrics["total_decoding_tokens"] / len(save_obj) if len(save_obj) > 0 else 0

    metrics["num_samples"] = len(save_obj)
    metrics["evaluated_num_samples"] = len(save_obj)
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
    metrics["requested_num_samples"] = num_samples
    metrics["dataset_block_index"] = dataset_block_index
    metrics["dataset_block_size"] = dataset_block_size
    metrics["evaluated_sample_start"] = evaluated_sample_start if save_obj else 0
    metrics["evaluated_sample_end"] = evaluated_sample_end if save_obj else 0
    metrics["max_new_tokens"] = max_new_tokens
    metrics["max_context_length"] = max_context_length
    metrics["random_seed"] = random_seed
    metrics["measure_memory"] = measure_memory
    metrics["measure_latency"] = measure_latency

    with open(str(score_filename), "w") as f:
        json.dump(metrics, f)
    print(metrics)
    return

if __name__ == "__main__":
    cache_dir = "/home/dixi/cache/"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    Fire(evaluate)
