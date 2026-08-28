# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Optional
from time import perf_counter

import torch
from datasets import load_dataset
from fire import Fire

from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
try:
    from seer_attn import SeerDecodingQwen2ForCausalLM
except ImportError:
    SeerDecodingQwen2ForCausalLM = None
try:
    from seer_attn import SeerDecodingQwen3ForCausalLM
except ImportError:
    SeerDecodingQwen3ForCausalLM = None
from kvpress import BasePress, KeyRerotationPress, PerLayerCompressionPress
from kvpress import ZipCachePress

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
from drop import drop_extractor, drop_formatter, drop_scorer
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
    RPCPress,
    SCOPEPress,
    H2OPress,
    SnapKVPress,
    PyramidKVPress,
    NonePress,
    TurboQuantPress,
)

logger = logging.getLogger(__name__)


def _tensor_storage_nbytes(value, device: torch.device, seen_storages: set) -> int:
    """Return unique storage bytes for tensors belonging to ``device``."""
    if not isinstance(value, torch.Tensor) or value.device != device:
        return 0
    try:
        storage = value.untyped_storage()
        storage_id = (value.device.type, value.device.index, storage.data_ptr())
        if storage_id in seen_storages:
            return 0
        seen_storages.add(storage_id)
        return storage.nbytes()
    except (AttributeError, RuntimeError):
        # Tensor subclasses do not always expose an untyped storage.
        storage_id = id(value)
        if storage_id in seen_storages:
            return 0
        seen_storages.add(storage_id)
        try:
            return value.nelement() * value.element_size()
        except (AttributeError, RuntimeError, TypeError):
            logger.debug("Unable to measure storage for tensor subclass %s", type(value).__name__)
            return 0


def cache_nbytes(cache, device: torch.device) -> int:
    """Measure tensor storage owned by a Transformers cache on one CUDA device.

    Cache implementations vary across Transformers releases (lists of tensors,
    DynamicLayer objects, and quantized cache objects). Walking the cache object
    makes this work for all of them while storage de-duplication avoids counting
    tensor views twice.
    """
    seen_objects = set()
    seen_storages = set()

    def visit(value) -> int:
        if isinstance(value, torch.Tensor):
            tensor_bytes = _tensor_storage_nbytes(value, device, seen_storages)
            if tensor_bytes or type(value) is torch.Tensor:
                return tensor_bytes
            # Wrapper tensor subclasses (for example quantized cache tensors)
            # may expose their payload only through Python attributes.
        if value is None or isinstance(value, (str, bytes, int, float, bool, torch.dtype, torch.device)):
            return 0

        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)

        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return sum(visit(item) for item in value)

        attributes = getattr(value, "__dict__", None)
        if attributes is None:
            return 0
        return sum(visit(item) for item in attributes.values())

    return visit(cache)


class GenerationPhaseTracker(LogitsProcessor):
    """Profile generation peaks and the actual KV-cache storage."""

    def __init__(self, device: str, measure_memory: bool, measure_latency: bool):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.measure_memory = measure_memory and self.device.type == "cuda"
        self.devices = [self.device] if self.device.type == "cuda" else []
        self.measure_latency = measure_latency
        self.prefill_memory_usage = 0.0
        self.decoding_memory_usage = 0.0
        self.baseline_memory_usage = 0.0
        self.peak_memory_usage = 0.0
        self.baseline_reserved_memory = 0.0
        self.prefill_reserved_memory = 0.0
        self.decoding_reserved_memory = 0.0
        self.peak_reserved_memory = 0.0
        self.prefill_cache_memory = 0.0
        self.peak_cache_memory = 0.0
        self.final_cache_memory = 0.0
        self.prefill_time = 0.0
        self.decoding_time = 0.0
        self._generation_start = None
        self._decoding_start = None
        self._model_hook = None
        self._prefill_cache_recorded = False

    def _synchronize(self):
        for profile_device in self.devices:
            torch.cuda.synchronize(profile_device)

    def _discover_model_devices(self, model):
        devices = {self.device}
        for entry in (getattr(model, "hf_device_map", None) or {}).values():
            if isinstance(entry, int):
                devices.add(torch.device("cuda", entry))
            elif isinstance(entry, torch.device) and entry.type == "cuda":
                resolved = entry
                if resolved.index is None:
                    resolved = torch.device("cuda", torch.cuda.current_device())
                devices.add(resolved)
            elif isinstance(entry, str) and (entry == "cuda" or entry.startswith("cuda:")):
                resolved = torch.device(entry)
                if resolved.index is None:
                    resolved = torch.device("cuda", torch.cuda.current_device())
                devices.add(resolved)
        self.devices = sorted(devices, key=lambda item: item.index or 0)

    def _sum_cuda_stat(self, stat) -> float:
        return sum(stat(profile_device) for profile_device in self.devices) / 1024**3

    def _reset_peak_stats(self):
        for profile_device in self.devices:
            torch.cuda.reset_peak_memory_stats(profile_device)

    def _record_cache(self, module, args, kwargs, output):
        cache = getattr(output, "past_key_values", None)
        if cache is None and isinstance(output, dict):
            cache = output.get("past_key_values")
        if cache is None:
            return

        cache_gb = sum(cache_nbytes(cache, profile_device) for profile_device in self.devices) / 1024**3
        if not self._prefill_cache_recorded:
            self.prefill_cache_memory = cache_gb
            self._prefill_cache_recorded = True
        self.final_cache_memory = cache_gb
        self.peak_cache_memory = max(self.peak_cache_memory, cache_gb)

    def start(self, model=None):
        """Start immediately before ``model.generate``."""
        if self.device.type == "cuda" and model is not None:
            self._discover_model_devices(model)
        if self.measure_memory or self.measure_latency:
            self._synchronize()
        if self.measure_memory:
            self.baseline_memory_usage = self._sum_cuda_stat(torch.cuda.memory_allocated)
            self.baseline_reserved_memory = self._sum_cuda_stat(torch.cuda.memory_reserved)
            self._reset_peak_stats()
            if model is not None:
                self._model_hook = model.register_forward_hook(self._record_cache, with_kwargs=True)
        self._generation_start = perf_counter()

    def __call__(self, input_ids, scores):
        # Transformers invokes logits processors after the initial forward pass
        # has produced its logits, which is the prefill/decode boundary.
        if self._decoding_start is None:
            if self.measure_memory or self.measure_latency:
                self._synchronize()
            boundary_time = perf_counter()
            if self.measure_latency:
                self.prefill_time = boundary_time - self._generation_start
            if self.measure_memory:
                self.prefill_memory_usage = self._sum_cuda_stat(torch.cuda.max_memory_allocated)
                self.prefill_reserved_memory = self._sum_cuda_stat(torch.cuda.max_memory_reserved)
                # The current allocation (model + compressed prompt cache) becomes
                # decoding's baseline; the earlier transient prefill peak is cleared.
                self._reset_peak_stats()
            self._decoding_start = boundary_time
        return scores

    def finish(self):
        """Finish immediately after ``model.generate`` and return phase metrics."""
        if self.measure_memory or self.measure_latency:
            self._synchronize()
        finish_time = perf_counter()

        # A logits processor should always run for generation. Keep a safe fallback
        # for unusual model implementations that return before applying processors.
        if self._decoding_start is None:
            self._decoding_start = finish_time
            if self.measure_latency:
                self.prefill_time = finish_time - self._generation_start
            if self.measure_memory:
                self.prefill_memory_usage = self._sum_cuda_stat(torch.cuda.max_memory_allocated)
                self.prefill_reserved_memory = self._sum_cuda_stat(torch.cuda.max_memory_reserved)
                self._reset_peak_stats()

        if self.measure_latency:
            self.decoding_time = finish_time - self._decoding_start
        if self.measure_memory:
            self.decoding_memory_usage = self._sum_cuda_stat(torch.cuda.max_memory_allocated)
            self.decoding_reserved_memory = self._sum_cuda_stat(torch.cuda.max_memory_reserved)
            self.peak_memory_usage = max(self.prefill_memory_usage, self.decoding_memory_usage)
            self.peak_reserved_memory = max(self.prefill_reserved_memory, self.decoding_reserved_memory)

        if self._model_hook is not None:
            self._model_hook.remove()
            self._model_hook = None

        return {
            "generation_phase_metrics_version": 2,
            "memory_profile_devices": [str(profile_device) for profile_device in self.devices],
            "prefill_memory_usage": self.prefill_memory_usage,
            "decoding_memory_usage": self.decoding_memory_usage,
            "memory_usage": self.peak_memory_usage,
            "baseline_memory_usage": self.baseline_memory_usage,
            "peak_memory_usage": self.peak_memory_usage,
            "peak_memory_above_baseline": max(0.0, self.peak_memory_usage - self.baseline_memory_usage),
            "baseline_reserved_memory": self.baseline_reserved_memory,
            "peak_reserved_memory": self.peak_reserved_memory,
            "peak_reserved_memory_above_baseline": max(
                0.0, self.peak_reserved_memory - self.baseline_reserved_memory
            ),
            "prefill_cache_memory": self.prefill_cache_memory,
            "peak_cache_memory": self.peak_cache_memory,
            "final_cache_memory": self.final_cache_memory,
            "prefill_time": self.prefill_time,
            "decoding_time": self.decoding_time,
            "total_time": self.prefill_time + self.decoding_time,
        }

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

# Accept the sweep/CLI spellings of dataset names that differ from the canonical
# DATASET_DICT keys. Without this, e.g. `--dataset=commonsense_qa` (as emitted by
# rerunall.sh) fails the assert below even though "commonsenseqa" is registered.
DATASET_ALIASES = {
    "commonsense_qa": "commonsenseqa",
    "csqa": "commonsenseqa",
    "openbook_qa": "openbookqa",
    "obqa": "openbookqa",
    "strategy_qa": "strategyqa",
    "math_500": "math500",
    "math-500": "math500",
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
    "drop": drop_extractor,
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

# Math-Verify performs its own answer extraction and is substantially more
# reliable when it can inspect the full response (for example, multiple LaTeX
# environments followed by a final boxed answer). Numeric scorers likewise
# prioritize explicit final-answer markers before falling back to the last
# numeric expression.
RAW_RESPONSE_SCORING_DATASETS = {"gsm8k", "aime24", "aime25", "math500"}

PRESS_DICT = {
    "knorm": KnormPress(),
    "h2o": H2OPress(),
    "random": RandomPress(),
    "streaming_llm": StreamingLLMPress(),
    "snapkv": SnapKVPress(),
    "snapkv_press": SnapKVPress(),  # Alias for snapkv
    "pyramidkv": PyramidKVPress(),
    "rkv": RKVPress(),
    "rkvlsh": RKVLSHPress(),
    "scope": SCOPEPress(),
    "rpc": RPCPress(),
    "full": FullPress(),
    "none": NonePress(),  # No-op press that does nothing
    "turboquant": TurboQuantPress(),
    "zipcache": ZipCachePress(),
}

POSITION_RETENTION_TRACKING_PRESSES = {
    "full",
    "h2o",
    "knorm",
    "random",
    "rkv",
    "rkvlsh",
    "snapkv",
    "snapkv_press",
    "streaming_llm",
    "scope",
    "rpc",
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
    lam: float = 0.1,
    n_bits: int = 4,
    zipcache_high_bits: int = 4,
    zipcache_low_bits: int = 2,
    zipcache_salient_ratio: float = 0.1,
    snapkv_window_size: int = 64,
    scope_decoding_cache_budget: int = 0,
    scope_compress_interval: int = 32,
    scope_decoding_window_size: int = 8,
    rpc_window_size: int = 32,
    rpc_compress_interval: int = 128,
    rpc_kernel_size: int = 7,
    track_tokens: bool = False,
    track_buckets: bool = False,
    enable_qualitative_analysis: bool = False,
    measure_memory: bool = True,
    measure_latency: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.9,
    run_tag: str = "",
    result_dir: Optional[str] = None,
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
    lam : float, optional
        Weight of attention importance in the RKV/RKV-LSH score; redundancy uses weight ``1 - lam``, by default 0.1
    enable_qualitative_analysis : bool, optional
        Enable qualitative token retention/eviction analysis for RKV-LSH, by default False
    snapkv_window_size : int, optional
        Recent/observation window size for H2O, SnapKV, and PyramidKV, by default 64
    scope_decoding_cache_budget : int, optional
        Decoding-phase cache budget for SCOPE (0 disables decoding-phase compression), by default 0
    scope_compress_interval : int, optional
        Number of newly generated tokens between SCOPE decoding-phase re-selections, by default 32
    scope_decoding_window_size : int, optional
        Number of most-recently generated tokens SCOPE never prunes during decoding, by default 8
    rpc_window_size : int, optional
        Selector/recent window size for RPC, by default 32
    rpc_compress_interval : int, optional
        Number of newly generated tokens between RPC re-selections (and the budget growth step), by default 128
    rpc_kernel_size : int, optional
        Pooling kernel size for RPC's importance scores, by default 7
    top_p : float, optional
        Top-p (nucleus) sampling parameter used when do_sampling is True, by default 0.9
    run_tag : str, optional
        Optional tag (e.g. "run1") appended to the output filename, useful for repeated runs with the same
        configuration/seed to measure sampling variance, by default "" (no tag)
    result_dir : str, optional
        Directory for result JSONL and score JSON files. Defaults to reason/results.
    measure_memory : bool, optional
        Whether to profile peak GPU allocation/reservation and actual KV-cache storage, by default True
    measure_latency : bool, optional
        Whether to measure separate full-phase prefill and decoding wall times, by default True
    """

    run_start = perf_counter()
    dataset_ingestion_time = 0.0
    model_load_time = 0.0
    evaluation_loop_time = 0.0
    scoring_time = 0.0
    execution_mode = "loaded_existing_results"

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

    dataset = DATASET_ALIASES.get(dataset, dataset)
    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    assert dataset in SCORER_DICT, f"No scorer found for {dataset}"
    assert dataset_block_index >= 0, "dataset_block_index must be >= 0"
    assert dataset_block_size > 0, "dataset_block_size must be > 0"

    hf_name = DATASET_DICT[dataset][0]
    data_dir = DATASET_DICT[dataset][1] if data_dir is None else data_dir
    data_split = DATASET_DICT[dataset][2] if data_split is None else data_split

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _default_tensor_device():
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _device_map_entry_to_device(device_entry):
        if isinstance(device_entry, int):
            return f"cuda:{device_entry}"
        if isinstance(device_entry, str):
            if device_entry == "disk":
                return None
            if device_entry.isdigit():
                return f"cuda:{device_entry}"
            return device_entry
        return None

    def _resolve_tensor_device(model=None):
        """Return a concrete torch device string for inputs and auxiliary tensors."""
        if device != "auto":
            return device

        if model is not None and hasattr(model, "hf_device_map"):
            for device_entry in model.hf_device_map.values():
                resolved = _device_map_entry_to_device(device_entry)
                if resolved is not None and resolved != "cpu":
                    return resolved
            for device_entry in model.hf_device_map.values():
                resolved = _device_map_entry_to_device(device_entry)
                if resolved is not None:
                    return resolved

        if model is not None and hasattr(model, "device"):
            model_device = str(model.device)
            if model_device != "meta":
                return model_device

        return _default_tensor_device()

    tensor_device = _resolve_tensor_device()

    save_dir = Path(result_dir).expanduser() if result_dir else Path(__file__).parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Result directory: {save_dir.resolve()}")
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
    elif press_name == "zipcache":
        save_filename = save_dir / (
            "__".join([dataset, data_dir if data_dir else "", model_name.replace("/", "--"), press_name, f"high{zipcache_high_bits}", f"low{zipcache_low_bits}", f"sal{int(round(zipcache_salient_ratio*100)):03d}", f"max_new_tokens{max_new_tokens}"])
            + ".jsonl"
        )
    elif press_name in ("h2o", "snapkv", "snapkv_press", "pyramidkv"):
        save_filename = save_dir / (
            "__".join([dataset, data_dir if data_dir else "", model_name.replace("/", "--"), press_name, f"budget{cache_budget}", f"window{snapkv_window_size}", f"max_new_tokens{max_new_tokens}"])
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
        top_p_sanitized = f"{top_p:.3f}".rstrip("0").rstrip(".").replace(".", "")
        save_filename = save_filename.with_name(
            save_filename.stem + f"__topp{top_p_sanitized}" + save_filename.suffix
        )
    if run_tag:
        save_filename = save_filename.with_name(save_filename.stem + f"__{run_tag}" + save_filename.suffix)
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
        execution_mode = "full_evaluation"
        # Open file for incremental writing (append mode)
        # Clear the file first if it exists
        if save_filename.exists():
            save_filename.unlink()
        # Load datasetf
        dataset_ingestion_start = perf_counter()
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
        dataset_ingestion_time = perf_counter() - dataset_ingestion_start

        # Load press
        if press_name not in PRESS_DICT:
            available_presses = ", ".join(sorted(PRESS_DICT))
            raise ValueError(f"Unknown press_name '{press_name}'. Available presses: {available_presses}")
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
        if press_name == "zipcache" and press is not None:
            press.high_bits = zipcache_high_bits
            press.low_bits = zipcache_low_bits
            press.salient_ratio = zipcache_salient_ratio
            # Budget is expressed through the bit mix, not by pruning tokens.
            press.cache_budget = 0

        if press_name in ("h2o", "snapkv", "snapkv_press", "pyramidkv") and press is not None:
            press.window_size = snapkv_window_size

        if press_name in ("rkv", "rkvlsh") and press is not None:
            press.lam = lam
        if press_name=="rkvlsh" and press is not None:
            press.n_hash_buckets=n_hash_buckets
            press.initialize_buckets(device=tensor_device)
            # Enable bucket tracking if requested
            if track_buckets:
                press.enable_bucket_tracking()

        if press_name == "scope" and press is not None:
            press.window_size = snapkv_window_size
            press.decoding_cache_budget = scope_decoding_cache_budget
            press.compress_interval = scope_compress_interval
            press.decoding_window_size = scope_decoding_window_size

        if press_name == "rpc" and press is not None:
            press.window_size = rpc_window_size
            press.compress_interval = rpc_compress_interval
            press.kernel_size = rpc_kernel_size

        if enable_qualitative_analysis:
            if press is None or not hasattr(press, "enable_qualitative_mode"):
                raise ValueError(
                    "--enable_qualitative_analysis is only supported by presses "
                    "that implement enable_qualitative_mode (currently rkv and rkvlsh)."
                )
            qualitative_filename = (
                f"{save_filename.stem}__token_decisions.jsonl"
            )
            press.enable_qualitative_mode(
                output_file=qualitative_filename,
                model_name=model_name.replace("/", "--"),
                press_name=press_name,
            )

        # Presses that consume attentions need eager attention.
        attention_config = {}
        if output_attentions(press):
            attention_config = {
                "output_attentions": True,
                "attn_implementation": "eager",
            }
        
        model_load_start = perf_counter()
        if "SeerAttention" in model_name:
            # `A or B` picks A whenever A is merely importable, so a Qwen3-based
            # SeerAttention checkpoint was silently loaded with the Qwen2 class.
            # Choose on the checkpoint's own architecture instead.
            _seer_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            _seer_arch = (getattr(_seer_cfg, "model_type", "") or "").lower()
            if "qwen3" in _seer_arch:
                seer_model_cls = SeerDecodingQwen3ForCausalLM or SeerDecodingQwen2ForCausalLM
            else:
                seer_model_cls = SeerDecodingQwen2ForCausalLM or SeerDecodingQwen3ForCausalLM
            if seer_model_cls is None:
                raise ImportError(
                    "SeerAttention model requested, but `seer_attn` is not installed "
                    "or does not export SeerDecodingQwen2ForCausalLM/SeerDecodingQwen3ForCausalLM."
                )
            # SeerAttention models: Load config first, then tokenizer from base_model
            # This is the recommended approach per SeerAttention documentation
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model,
                trust_remote_code=True,
                padding_side="left",
            )
            model = seer_model_cls.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                seerattn_sparsity_method='token_budget',
                seerattn_token_budget=cache_budget,
                **attention_config
            )
            model.to(tensor_device)
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
            tensor_device = _resolve_tensor_device(model)
        model_load_time = perf_counter() - model_load_start
        
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
        evaluation_loop_start = perf_counter()
        for i, example in tqdm(enumerate(ds), total=len(ds)):
            # Aggressive memory cleanup at the START of each sample to prevent accumulation
            import gc
            torch.cuda.empty_cache()
            if measure_memory:  # Only synchronize if measuring memory
                torch.cuda.synchronize()
            gc.collect()
            
            input_text, gt_answer_text = formatter(example)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(tensor_device)
            if max_context_length is not None:
                inputs = {k: v[:, :max_context_length] for k, v in inputs.items()}
            if max_new_tokens is None:
                max_new_tokens = 16 * 1024 - inputs["input_ids"].shape[1] # use 16k for max length for now

            # Clear unused cached blocks before generation. Phase-local peak stats
            # are reset by GenerationPhaseTracker immediately before each phase.
            if measure_memory:
                torch.cuda.empty_cache()

            phase_tracker = GenerationPhaseTracker(
                device=tensor_device,
                measure_memory=measure_memory,
                measure_latency=measure_latency,
            )
            phase_logits_processors = LogitsProcessorList([phase_tracker])
            sampling_kwargs = {
                "do_sample": do_sampling,
            }
            if do_sampling:
                sampling_kwargs.update(
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=1.2,
                )

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
                
                phase_tracker.start(model)
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    output_attentions=False,
                    logits_processor=phase_logits_processors,
                    **sampling_kwargs,
                )
                phase_metrics = phase_tracker.finish()
                
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
                
                with press_context:
                    phase_tracker.start(model)
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        output_attentions=output_attentions(press) if press is not None and not isinstance(press, NonePress) else False,
                        logits_processor=phase_logits_processors,
                        **sampling_kwargs,
                    )
                    phase_metrics = phase_tracker.finish()
                
                # Decode response for standard path
                pred_start = inputs["input_ids"].shape[1]
                response = tokenizer.decode(outputs[0][pred_start:], skip_special_tokens=True)
                # Clean up special token artifacts (\u0120 = Ġ, \u010a = Ċ, etc.)
                response = response.replace("\u0120", " ").replace("\u010a", " ").replace("\u0109", " ")
                # Remove other control characters while preserving spaces and newlines
                response = "".join(c if ord(c) >= 32 or c in "\n\r\t" else "" for c in response)
                model_answer = extractor(response)

            # Get timing metrics from press if available (before deleting tensors)
            press_timing_metrics = {}
            if press is not None and hasattr(press, 'get_timing_metrics'):
                press_timing_metrics = {
                    f"press_{key}": value
                    for key, value in press.get_timing_metrics().items()
                }

            # Calculate metrics before deleting tensors
            input_token_count = inputs["input_ids"].shape[1]
            output_token_count = outputs[0].shape[0] - input_token_count
            response_token_length = output_token_count
            total_token_count = outputs[0].shape[0]
            sequence_token_ids = outputs[0].detach().cpu().tolist() if track_tokens else []
            
            phase_metrics["output_tokens_per_second"] = (
                output_token_count / phase_metrics["decoding_time"]
                if measure_latency and phase_metrics["decoding_time"] > 0
                else 0.0
            )
            phase_metrics["total_prefill_tokens"] = input_token_count
            phase_metrics["total_decoding_tokens"] = output_token_count
            
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
                    "memory_usage": phase_metrics["memory_usage"],
                    "prefill_memory_usage": phase_metrics["prefill_memory_usage"],
                    "decoding_memory_usage": phase_metrics["decoding_memory_usage"],
                    "baseline_memory_usage": phase_metrics["baseline_memory_usage"],
                    "peak_memory_usage": phase_metrics["peak_memory_usage"],
                    "peak_memory_above_baseline": phase_metrics["peak_memory_above_baseline"],
                    "baseline_reserved_memory": phase_metrics["baseline_reserved_memory"],
                    "peak_reserved_memory": phase_metrics["peak_reserved_memory"],
                    "peak_reserved_memory_above_baseline": phase_metrics[
                        "peak_reserved_memory_above_baseline"
                    ],
                    "prefill_cache_memory": phase_metrics["prefill_cache_memory"],
                    "peak_cache_memory": phase_metrics["peak_cache_memory"],
                    "final_cache_memory": phase_metrics["final_cache_memory"],
                    "execution_time": phase_metrics["total_time"],
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
            
            # Phase metrics cover the full model.generate call. Press timing metrics
            # cover only the compression hooks and use an explicit prefix.
            save_obj.update(phase_metrics)
            save_obj.update(press_timing_metrics)
            
            if track_tokens and not is_seer_attention_none:
                # Track keyword retention if press tracks retention
                keyword_retention = {}
                final_retained_indices = []
                tracked_sequence_length = 0
                retention_tracking_status = "unavailable"
                if (
                    press is not None
                    and not isinstance(press, NonePress)
                    and press_name in POSITION_RETENTION_TRACKING_PRESSES
                    and hasattr(press, "retention_tracking_is_reliable")
                    and press.retention_tracking_is_reliable()
                ):
                    final_retained_indices = list(press.get_final_retained_indices())
                    tracked_sequence_length = press.get_tracked_sequence_length()
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
                    retention_tracking_status = "tracked"
                elif press is None or isinstance(press, NonePress):
                    # No compression means every input position is genuinely retained.
                    tracked_sequence_length = max(total_token_count - 1, input_token_count)
                    final_retained_indices = list(range(tracked_sequence_length))
                    retention_results = track_token_retention(
                        input_token_ids,
                        list(range(len(input_token_ids))),
                        keyword_token_ids,
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
                    retention_tracking_status = "no_compression"
                else:
                    retention_tracking_status = "unsupported_for_press"
                
                # Add keyword retention to save_obj
                save_obj['keywords'] = keywords
                save_obj['keyword_retention'] = keyword_retention
                save_obj['retention_tracking_status'] = retention_tracking_status
                save_obj['input_token_ids'] = input_token_ids
                save_obj['sequence_token_ids'] = sequence_token_ids
                save_obj['final_retained_indices'] = sorted(final_retained_indices)
                save_obj['tracked_sequence_length'] = tracked_sequence_length
                save_obj['retention_tracking_scope'] = "layer_0_kv_head_0"
                
                generation_steps = []
                if not is_seer_attention_none and press is not None and not isinstance(press, NonePress) and hasattr(press, 'get_generation_steps'):
                    generation_steps = press.get_generation_steps()
                
                # Save generation_steps to save_obj
                save_obj['generation_steps'] = generation_steps
            else:
                save_obj['keywords'] = {}
                save_obj['keyword_retention'] = {}
                save_obj['generation_steps'] = []
                save_obj['retention_tracking_status'] = "disabled"
            
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

            print(
                f"✅ [{i+1}/{len(ds)}] Saved result for question {i+1} to {save_filename.name} "
                f"(GPU peak above baseline: {phase_metrics['peak_memory_above_baseline']:.2f} GB, "
                f"peak KV cache: {phase_metrics['peak_cache_memory']:.2f} GB)"
            )
            
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

        evaluation_loop_time = perf_counter() - evaluation_loop_start
        print(f"\n✅ All results saved to {save_filename}")
    # end of the if save_filename.exists()

    # load the results and evaluate the metrics
    with open(str(save_filename), "r") as f:
        save_obj = [json.loads(line) for line in f.readlines()]
    extracted_answers = [
        obj.get("response", obj["extracted_answer"])
        if dataset in RAW_RESPONSE_SCORING_DATASETS
        else obj["extracted_answer"]
        for obj in save_obj
    ]
    gt_answers = [obj["gt_answer"] for obj in save_obj]

    if save_obj:
        if dataset_block_index > 0:
            evaluated_sample_start = (dataset_block_index - 1) * dataset_block_size + 1
            evaluated_sample_end = evaluated_sample_start + len(save_obj) - 1
        else:
            evaluated_sample_start = 1
            evaluated_sample_end = len(save_obj)

    # Calculate metrics
    scoring_start = perf_counter()
    scorer = SCORER_DICT[dataset]
    metrics = scorer(extracted_answers, gt_answers)
    scoring_time = perf_counter() - scoring_start

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
        if all(obj.get("generation_phase_metrics_version", 0) >= 1 for obj in save_obj):
            metrics["avg_prefill_memory_usage_gb"] = sum(
                obj["prefill_memory_usage"] for obj in save_obj
            ) / len(save_obj)
            metrics["max_prefill_memory_usage_gb"] = max(
                obj["prefill_memory_usage"] for obj in save_obj
            )
            metrics["avg_decoding_memory_usage_gb"] = sum(
                obj["decoding_memory_usage"] for obj in save_obj
            ) / len(save_obj)
            metrics["max_decoding_memory_usage_gb"] = max(
                obj["decoding_memory_usage"] for obj in save_obj
            )
        if all(obj.get("generation_phase_metrics_version", 0) >= 2 for obj in save_obj):
            metrics["avg_baseline_memory_usage_gb"] = sum(
                obj["baseline_memory_usage"] for obj in save_obj
            ) / len(save_obj)
            metrics["peak_gpu_memory_usage_gb"] = max(obj["peak_memory_usage"] for obj in save_obj)
            metrics["peak_gpu_memory_above_baseline_gb"] = max(
                obj["peak_memory_above_baseline"] for obj in save_obj
            )
            metrics["peak_gpu_reserved_memory_gb"] = max(
                obj["peak_reserved_memory"] for obj in save_obj
            )
            metrics["peak_gpu_reserved_memory_above_baseline_gb"] = max(
                obj["peak_reserved_memory_above_baseline"] for obj in save_obj
            )
            metrics["avg_prefill_cache_memory_gb"] = sum(
                obj["prefill_cache_memory"] for obj in save_obj
            ) / len(save_obj)
            metrics["max_prefill_cache_memory_gb"] = max(
                obj["prefill_cache_memory"] for obj in save_obj
            )
            metrics["avg_peak_cache_memory_gb"] = sum(
                obj["peak_cache_memory"] for obj in save_obj
            ) / len(save_obj)
            metrics["peak_cache_memory_gb"] = max(obj["peak_cache_memory"] for obj in save_obj)
            metrics["avg_final_cache_memory_gb"] = sum(
                obj["final_cache_memory"] for obj in save_obj
            ) / len(save_obj)
            metrics["max_final_cache_memory_gb"] = max(
                obj["final_cache_memory"] for obj in save_obj
            )
    
    # Add latency metrics if measured
    if measure_latency and save_obj:
        metrics["avg_execution_time"] = sum([obj.get("execution_time", 0.0) for obj in save_obj]) / len(save_obj)
        metrics["total_execution_time"] = sum([obj.get("execution_time", 0.0) for obj in save_obj])
    
    # Add full-generation phase timing averages.
    if (
        measure_latency
        and save_obj
        and all(obj.get("generation_phase_metrics_version", 0) >= 1 for obj in save_obj)
    ):
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
    metrics["temperature"] = temperature
    metrics["top_p"] = top_p
    metrics["run_tag"] = run_tag
    if press_name in ("h2o", "snapkv", "snapkv_press", "pyramidkv"):
        metrics["window_size"] = snapkv_window_size
    if press_name in ("rkv", "rkvlsh"):
        metrics["lam"] = lam
    if press_name=="rkvlsh":
        metrics["n_hash_buckets"] = n_hash_buckets
    if press_name == "scope":
        metrics["scope_decoding_cache_budget"] = scope_decoding_cache_budget
        metrics["scope_compress_interval"] = scope_compress_interval
        metrics["scope_decoding_window_size"] = scope_decoding_window_size
    if press_name == "rpc":
        metrics["rpc_window_size"] = rpc_window_size
        metrics["rpc_compress_interval"] = rpc_compress_interval
        metrics["rpc_kernel_size"] = rpc_kernel_size
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
    metrics["result_dir"] = str(save_dir)
    if measure_memory:
        metrics["memory_profile_version"] = 1
        metrics["memory_profile_scope"] = "all_model_cuda_devices"
        metrics["memory_profile_devices"] = (
            save_obj[0].get("memory_profile_devices", [str(torch.device(tensor_device))])
            if save_obj
            else [str(torch.device(tensor_device))]
        )
        metrics["memory_profile_units"] = "GiB"
    if save_obj and all(obj.get("generation_phase_metrics_version", 0) >= 1 for obj in save_obj):
        metrics["generation_phase_metrics_version"] = min(
            obj["generation_phase_metrics_version"] for obj in save_obj
        )

    # Run-level wall times deliberately include work outside model.generate.
    if measure_memory or measure_latency:
        metrics["timing_scope"] = "evaluate_entry_through_scoring"
        metrics["execution_mode"] = execution_mode
        metrics["dataset_ingestion_time_seconds"] = dataset_ingestion_time
        metrics["model_load_time_seconds"] = model_load_time
        metrics["evaluation_loop_time_seconds"] = evaluation_loop_time
        metrics["scoring_time_seconds"] = scoring_time
        metrics["end_to_end_time_seconds"] = perf_counter() - run_start

    with open(str(score_filename), "w") as f:
        json.dump(metrics, f)
    print(metrics)
    return

if __name__ == "__main__":
    cache_dir = "/home/dixi/cache/"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    Fire(evaluate)
