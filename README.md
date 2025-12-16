[![PyPI version](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab example notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/nvidia/kvpress)
[![Blog post](https://img.shields.io/badge/🤗%20Hugging%20Face-Blog-blue)](https://huggingface.co/blog/nvidia/kvpress)

![kvpress](kvpress.jpg)

## Overview
Long-context inference is dominated by the cost of storing key-value (KV) caches whose size grows with the number of processed tokens. NVIDIA's original **kvpress** library introduced a consistent interface for "presses" that prune those caches during prefill. This fork keeps the public API intact while adding the features required for our paper **Hold Onto That Thought: Assessing KV Cache Compression on Reasoning**:
- Every press can now act during decoding, which matters for reasoning traces that often exceed the prompt length.
- Timing hooks, token counters, and CUDA memory statistics were added so we can compare latency and accuracy under tight cache budgets.
- A new reasoning benchmark harness (`reason/`) covers eight public datasets plus AIME24/25, mirroring the study in the paper.
- The LaTeX source for the paper itself lives in `KV_Compression_Reasoning_Eval 2/` for transparency and reproducibility.

## What changed in this fork?
- **Decoding-aware presses.** `BasePress` gained `compress_decoding`, latency tracking, and CSV logging. This enables H2O, SnapKV-D, StreamingLLM, K-Norm, R-KV, ShadowKV, and the other 20+ presses to manage cache budgets throughout generation rather than only during prefill.
- **`KVPressTextGenerationPipeline`.** The pipeline registers every press as `kv-press-text-generation`, handles chat templates, and exposes controls for `max_new_tokens`, context truncation, and cache reuse for multiple questions.
- **Reasoning evaluation harness.** The `reason/evaluate.py` CLI loads GSM8K, MATH-500, FOLIO, DROP, StrategyQA, ReClor, CommonsenseQA, OpenBookQA, LogiQA, AIME24, and AIME25 from Hugging Face, formats prompts, and logs results to JSON with timing and compression ratios.
- **Cluster scripts and notebooks.** `reason/run_experiments.sh` reproduces the sweep over cache budgets {128, 256, 384, 512} and 2k decoding limits. The original long-context benchmarks (`evaluation/`) and demo notebooks are kept untouched.

## Findings from *Hold Onto That Thought*
The full discussion is in the paper (source under `KV_Compression_Reasoning_Eval 2/`). A few practical takeaways:
1. Llama-3.1-8B-Instruct does not have a single best press. H2O and StreamingLLM tend to do well on reading comprehension, while SnapKV-D is stronger on shorter prompts with long chains of thought.
2. On reasoning-oriented models (Llama-3.1-Nemotron-Nano-8B-v1, DeepSeek-R1-Distill Qwen/Llama variants), attention-based heavy-hitter tracking dominates: H2O and SnapKV-D frequently match or exceed the uncompressed baseline even at 256-token budgets.
3. At low budgets, cosine-similarity pruning (R-KV, K-Norm) can lengthen reasoning traces because they discard redundant intermediate steps, exposing a trade-off between cache size and total decoding cost.

## Installation
### PyPI release
```bash
pip install kvpress
```
This gives you the latest official NVIDIA release. It does not yet include the reasoning harness or decoding hooks.

### From source (needed for the paper results)
```bash
git clone https://github.com/minghui-liu/kvpress.git
cd kvpress
pip install -e .
```
FlashAttention improves throughput for most models:
```bash
pip install flash-attn --no-build-isolation
```

## Using a press in Python
```python
from transformers import pipeline
from kvpress import H2OPress

model = "meta-llama/Llama-3.1-8B-Instruct"
model_kwargs = {"attn_implementation": "flash_attention_2"}
press = H2OPress(cache_budget=256)
press.latency = True  # collect timing stats

pipe = pipeline(
    "kv-press-text-generation",
    model=model,
    device="cuda:0",
    model_kwargs=model_kwargs,
)

context = "...long document..."
question = "Summarize the terms in plain English."
output = pipe(context, question=question, press=press, max_new_tokens=512)
print(output["answer"])
print(press.get_timing_metrics())
```
Tips:
- Provide `questions=[...]` to reuse the compressed context across multiple queries.
- Presses that need raw attention weights (`ObservedAttentionPress`, some research prototypes) require `model_kwargs={"attn_implementation": "eager"}`.
- You can wrap `with press(model): outputs = model.generate(...)` if you prefer direct `generate` calls, understanding that the question tokens cannot be excluded from compression there.

## Reasoning benchmark CLI
```
cd reason
python evaluate.py \
  --dataset gsm8k \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --press_name h2o \
  --cache_budget 256 \
  --max_new_tokens 2048 \
  --num_samples 100 \
  --device cuda:0
```
Important flags:
- `--press_name`: one of `knorm`, `h2o`, `random`, `streaming_llm`, `rkv`, `rkv_lsh`, or `full`. Add new presses to `PRESS_DICT` in `reason/evaluate.py`.
- `--cache_budget`: target sequence length per head after pruning.
- `--fraction` or `--num_samples`: subsample when debugging.
- `--max_context_length` / `--max_new_tokens`: defaults follow the paper (2k decoding tokens).
Each run writes prompts, responses, metrics, and telemetry to `reason/results/<dataset>__...json`. `reason/README.md` documents formatter/score snippets, and `reason/run_experiments.sh` shows the exact sweep used in the paper.

## Repository guide
- `kvpress/`: library code (presses, pipeline, attention patch, instrumentation).
- `reason/`: reasoning benchmark CLI and dataset scripts.
- `evaluation/`: original long-context benchmarks (LongBench, RULER, InfiniteBench, etc.).
- `notebooks/`: demos including the Wikipedia compression walkthrough.
- `tests/`: unit tests for selected presses.
- `KV_Compression_Reasoning_Eval 2/`: LaTeX for the paper.

## Press catalog (selected examples)
- **H2OPress** – accumulates per-head attention weights and keeps heavy hitters; good default on reasoning workloads.
- **SnapKVPress (SnapKV-D)** – sliding-window attention estimates extended to decoding; strong on math datasets.
- **StreamingLLMPress** – maintains an initial sink plus a moving window and is predictable when you must bound memory tightly.
- **KnormPress** – keeps tokens with large key norms; simple baseline that sometimes excels on GSM8K/MATH-500.
- **RKVPress / RKVLSHPress** – cosine-similarity eviction with optional locality-sensitive hashing to remove redundant states.
- **ObservedAttentionPress, ExpectedAttentionPress, QFilterPress, ChunkKVPress, ThinKPress, PyramidKVPress, FinchPress, RandomPress, FullPress** – additional strategies replicated from prior work. All inherit from `BasePress`, so they automatically gain decoding support and instrumentation.
To implement your own method, subclass `BasePress` or `ScorerPress`, override `compress_prefilling`/`compress_decoding`, and register the forward hook when evaluating.

## FAQ
- **Which attention backend should I use?** FlashAttention (`model_kwargs={"attn_implementation": "flash_attention_2"}`) for most presses. Switch to eager attention only when a method needs materialized attention matrices.
- **How do I run across multiple GPUs?** Let `pipeline(..., device_map="auto")` shard the model or use Accelerate for more control. The benchmark scripts assume a single GPU per job but can be adapted.
- **How do I inspect throughput changes?** Set `press.latency = True` before running; `press.get_timing_metrics()` reports prefill/decoding times and token counts. The reasoning CLI also logs CUDA memory stats gathered via `torch.cuda.memory_stats()`.
- **Can I disable decoding compression?** Set `press.cache_budget = 0` before generation to fall back to the full cache for comparison.

## Citation
If you use this repository, please cite both the original kvpress project and our paper:
- `CITATION.cff` (NVIDIA kvpress).
- Minghui Liu*, Aadi Palnitkar*, Tahseen Rabbani*, et al. *Hold Onto That Thought: Assessing KV Cache Compression on Reasoning*, 2025. (Source in `KV_Compression_Reasoning_Eval 2/`).
