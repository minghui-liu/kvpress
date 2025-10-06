# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import logging
import os
from pathlib import Path
import signal
from typing import Optional

import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from kvpress import BasePress, KeyRerotationPress, PerLayerCompressionPress
from time import time
from utils import default_extractor
from gsm8k import gsm8k_formatter, gsm8k_scorer
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
    "rkv": RKVPress(),
    "rkv_lsh": RKVLSHPress(),
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

def collect_cuda_memory_metrics():
    stats = torch.cuda.memory_stats()
    return {
        "allocated_current": torch.cuda.memory_allocated(),
        "allocated_peak": torch.cuda.max_memory_allocated(),
        "reserved_current": torch.cuda.memory_reserved(),
        "reserved_peak": torch.cuda.max_memory_reserved(),
        "fragmentation": float(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / max(1, torch.cuda.memory_reserved()),
        "inactive_split_bytes": stats.get("inactive_split_bytes.all.current", 0),
        "segments": stats.get("segment.all.current", 0),
        "allocations": stats.get("allocation.all.current", 0),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
        "pinned_current": stats.get("active_bytes.pinned.current", 0),
        "pinned_peak": stats.get("active_bytes.pinned.peak", 0),
    }

def reset_cuda_stats():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


class StopOnRepetition(StoppingCriteria):
    def __init__(self, prompt_len: int, min_repeat: int = 8, ngram_min: int = 2, ngram_max: int = 15, window: int = 300, tokenizer=None):
        super().__init__()
        self.prompt_len = prompt_len
        self.min_repeat = min_repeat
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.window = window
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):  # noqa: D401, ARG002
        try:
            seq = input_ids[0].tolist()
            gen_ids = seq[self.prompt_len:]
            if len(gen_ids) < self.ngram_min * self.min_repeat:
                return False
            start_idx = max(0, len(gen_ids) - self.window)
            tail = gen_ids[start_idx:]
            # Detect any n-gram that repeats consecutively >= min_repeat times
            for n in range(self.ngram_min, self.ngram_max + 1):
                limit = len(tail) - n * self.min_repeat + 1
                if limit <= 0:
                    continue
                for i in range(limit):
                    seg = tail[i:i + n]
                    repeated = True
                    for r in range(1, self.min_repeat):
                        a = i + r * n
                        b = a + n
                        if tail[a:b] != seg:
                            repeated = False
                            break
                    if repeated:
                        try:
                            snippet = ""
                            if self.tokenizer is not None:
                                snippet = self.tokenizer.decode(seg, skip_special_tokens=True)
                            print(f"[STOP] repetition detected: n={n} x{self.min_repeat} snippet='{snippet[:80]}'")
                        except Exception:
                            print(f"[STOP] repetition detected: n={n} x{self.min_repeat}")
                        return True
            return False
        except Exception:
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
    do_sampling: bool = False,
    skip_existing: bool = True,
    compression_ratio: float = 0.1,
    key_channel_compression_ratio: float = 0.5,
    debug: bool = False,
    latency: bool = False,
    resume: bool = False,
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
    # Always include the random seed in filenames
    save_filename = save_filename.with_name(save_filename.stem + f"__seed{random_seed}" + save_filename.suffix)
    score_filename = save_dir / (save_filename.stem + "_score.json")

    if skip_existing and score_filename.exists():
        logger.warning(f"Score file already exists at {score_filename}, skipping evaluation")
        return

    if skip_existing and save_filename.exists():
        logger.warning(f"Model responses already exist at {save_filename}")
        print(f"Model responses already exist. Loading responses from {save_filename} and evaluating metrics")
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

        # Forward debug/latency flags to the press
        press.debug = debug
        press.latency = latency

        # Base CSV path for attention-loss/debug rows under results/csvs
        csv_dir = save_dir / "csvs"
        csv_dir.mkdir(exist_ok=True)
        base_csv_path = csv_dir / save_filename.with_suffix(".csv").name
        press.csv_path = str(base_csv_path)

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
        # Ensure the model has a valid pad_token_id configured
        try:
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass

        # Run generation on each context of the dataset
        save_objs = []

        # If requested, continue from the latest partial results
        start_index = 0
        if resume:
            try:
                saved_dir = save_dir / "saved"
                prefix = save_filename.stem + "__partial_n"
                latest_n = -1
                latest_path = None
                if saved_dir.exists():
                    for name in os.listdir(saved_dir):
                        if name.startswith(prefix) and name.endswith(".jsonl"):
                            try:
                                n_str = name[len(prefix):-6]  # strip prefix and .jsonl
                                n_val = int(n_str)
                                if n_val > latest_n:
                                    latest_n = n_val
                                    latest_path = saved_dir / name
                            except Exception:
                                pass
                if latest_path is not None and latest_n > 0:
                    with open(str(latest_path), "r") as f_:
                        save_objs = [json.loads(line) for line in f_.read().splitlines() if line]
                    start_index = min(latest_n, len(ds))
                    print(f"[RESUME] Resuming from partial results ({start_index} samples) at {latest_path}")
            except Exception:
                pass

        pbar = tqdm(total=len(ds), bar_format="{l_bar}{bar}{r_bar}")
        if start_index > 0:
            try:
                pbar.n = start_index
                pbar.refresh()
            except Exception:
                pass
        pbar.set_postfix({"dec": 0}, refresh=True)

        # Graceful early-exit handler to persist partial results under results/saved
        def _dump_partial_results():
            try:
                if not save_objs:
                    return
                saved_dir = save_dir / "saved"
                saved_dir.mkdir(exist_ok=True)
                num_done = len(save_objs)
                partial_path = saved_dir / (save_filename.stem + f"__partial_n{num_done}.jsonl")
                with open(str(partial_path), "w") as f_:
                    for obj in save_objs:
                        f_.write(json.dumps(obj) + "\n")
                print(f"[EARLY-SAVE] Partial results saved to {partial_path}")
            except Exception as _:
                pass

        def _sig_handler(signum, frame):  # noqa: ARG001
            try:
                _dump_partial_results()
            finally:
                try:
                    pbar.close()
                except Exception:
                    pass
                os._exit(1)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        for i, example in enumerate(ds):
            # Skip already processed samples when continuing
            if i < start_index:
                continue
            # Differentiate CSV per-sample
            press.csv_path = str(base_csv_path.with_name(base_csv_path.stem + f"__sample{i}" + base_csv_path.suffix))
            # Overwrite per-sample CSV if it already exists (one CSV per run)
            try:
                if os.path.exists(press.csv_path):
                    os.remove(press.csv_path)
            except Exception:
                pass
            input_text, gt_answer_text = formatter(example)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
            if max_context_length is not None:
                inputs = {k: v[:, :max_context_length] for k, v in inputs.items()}
            if max_new_tokens is None:
                max_new_tokens = 16 * 1024 - inputs["input_ids"].shape[1] # use 16k for max length for now
            
            if latency:
                reset_cuda_stats()
                if press is not None:
                    press.reset_timing()
                start = time()

            # Wire a progress callback from the press to update the tqdm postfix
            # Ensure per-token updates and enable progress reporting
            setattr(press, "progress_print_every", 1)
            setattr(press, "progress_enabled", True)

            def _progress_update(prefill_tokens: int, decoding_tokens: int, phase: str, label: str):
                try:
                    pbar.set_postfix({"dec": decoding_tokens}, refresh=True)
                    # Do not advance pbar here; it's the sample-level bar
                except Exception:
                    pass
            setattr(press, "progress_update", _progress_update)

            # Run generation
            try:
                pred_start = inputs["input_ids"].shape[1]
                stopping = StoppingCriteriaList([StopOnRepetition(prompt_len=pred_start, min_repeat=5, ngram_min=2, ngram_max=10, window=200, tokenizer=tokenizer)])
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
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping,
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
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping,
                            use_cache=True,
                            output_attentions=output_attentions(press),
                        )
            except KeyboardInterrupt:
                _dump_partial_results()
                pbar.close()
                return

            pred_start = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][pred_start:], skip_special_tokens=True)
            model_answer = extractor(response)

            if latency:
                torch.cuda.synchronize()
                stats = collect_cuda_memory_metrics()
                execution_time = time() - start
                timing_metrics = press.get_timing_metrics() if press is not None else {}
                # Prepare cold-cache for the next iteration (outside measured window)
                reset_cuda_stats()



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

            if latency:
                save_obj.update(timing_metrics)
                save_obj.update(stats)
                save_obj["execution_time"] = execution_time
            
            save_objs.append(save_obj)
            # Advance the per-sample progress bar and reset decoding token postfix for next sample
            try:
                pbar.update(1)
                pbar.set_postfix({"dec": 0}, refresh=True)
            except Exception:
                pass

        pbar.close()
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

    if latency and len(save_obj) > 0:
        # Aggregate latency metrics across samples
        exec_times = [o.get("execution_time") for o in save_obj if o.get("execution_time") is not None]
        prefill_times = [o.get("prefill_time") for o in save_obj if o.get("prefill_time") is not None]
        decoding_times = [o.get("decoding_time") for o in save_obj if o.get("decoding_time") is not None]
        press_tps = [o.get("output_tokens_per_second") for o in save_obj if o.get("output_tokens_per_second") is not None]
        end_to_end_tps = []
        for o in save_obj:
            if o.get("execution_time") is not None and o.get("output_token_count") is not None and o["execution_time"] > 0:
                end_to_end_tps.append(o["output_token_count"] / o["execution_time"])

        def _avg(vals):
            vals = [v for v in vals if v is not None]
            return float(sum(vals) / len(vals)) if vals else None

        # Average token counters from press timing metrics (saved per-sample when latency=True)
        total_prefill_tokens_list = [o.get("total_prefill_tokens") for o in save_obj if o.get("total_prefill_tokens") is not None]
        total_decoding_tokens_list = [o.get("total_decoding_tokens") for o in save_obj if o.get("total_decoding_tokens") is not None]

        # Average CUDA memory stats across samples
        allocated_current_vals = [o.get("allocated_current") for o in save_obj if o.get("allocated_current") is not None]
        allocated_peak_vals = [o.get("allocated_peak") for o in save_obj if o.get("allocated_peak") is not None]
        reserved_current_vals = [o.get("reserved_current") for o in save_obj if o.get("reserved_current") is not None]
        reserved_peak_vals = [o.get("reserved_peak") for o in save_obj if o.get("reserved_peak") is not None]
        fragmentation_vals = [o.get("fragmentation") for o in save_obj if o.get("fragmentation") is not None]
        inactive_split_bytes_vals = [o.get("inactive_split_bytes") for o in save_obj if o.get("inactive_split_bytes") is not None]
        segments_vals = [o.get("segments") for o in save_obj if o.get("segments") is not None]
        allocations_vals = [o.get("allocations") for o in save_obj if o.get("allocations") is not None]
        num_alloc_retries_vals = [o.get("num_alloc_retries") for o in save_obj if o.get("num_alloc_retries") is not None]
        num_ooms_vals = [o.get("num_ooms") for o in save_obj if o.get("num_ooms") is not None]
        pinned_current_vals = [o.get("pinned_current") for o in save_obj if o.get("pinned_current") is not None]
        pinned_peak_vals = [o.get("pinned_peak") for o in save_obj if o.get("pinned_peak") is not None]

        metrics.update({
            "avg_execution_time": _avg(exec_times),
            "avg_prefill_time": _avg(prefill_times),
            "avg_decoding_time": _avg(decoding_times),
            "avg_press_tokens_per_second": _avg(press_tps),
            "avg_end_to_end_tokens_per_second": _avg(end_to_end_tps),
            # Token counter averages
            "avg_total_prefill_tokens": _avg(total_prefill_tokens_list),
            "avg_total_decoding_tokens": _avg(total_decoding_tokens_list),
            # CUDA memory averages
            "avg_allocated_current": _avg(allocated_current_vals),
            "avg_allocated_peak": _avg(allocated_peak_vals),
            "avg_reserved_current": _avg(reserved_current_vals),
            "avg_reserved_peak": _avg(reserved_peak_vals),
            "avg_fragmentation": _avg(fragmentation_vals),
            "avg_inactive_split_bytes": _avg(inactive_split_bytes_vals),
            "avg_segments": _avg(segments_vals),
            "avg_allocations": _avg(allocations_vals),
            "avg_num_alloc_retries": _avg(num_alloc_retries_vals),
            "avg_num_ooms": _avg(num_ooms_vals),
            "avg_pinned_current": _avg(pinned_current_vals),
            "avg_pinned_peak": _avg(pinned_peak_vals),
        })

    with open(str(score_filename), "w") as f:
        json.dump(metrics, f)
    print(metrics)
    return

if __name__ == "__main__":
    cache_dir = "/fs/nexus-scratch/minghui/.cache/huggingface"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    Fire(evaluate)
