## Installation

```bash
pip install kvpress
```

If possible, install flash attention:
```bash
pip install flash-attn --no-build-isolation
```

## Available presses

All current presses are training free and inherit from `BasePress` ([source](kvpress/presses/base_press.py)). 

Several presses inherit from `ScorerPress` ([source](kvpress/presses/scorer_press.py)) and rely on a score to prune the KV pairs with lowest importance:

- `RandomPress` ([source](kvpress/presses/random_press.py)): random score
- `KnormPress` ([source](kvpress/presses/knorm_press.py), [paper](https://arxiv.org/abs/2406.11430)): inverse norm of the key
- `SnapKVPress` ([source](kvpress/presses/snapkv_press.py), [paper](https://arxiv.org/abs/2404.14469)): average attention weight of the last queries
- `ExpectedAttentionPress` ([source](kvpress/presses/expected_attention_press.py), [notebook](notebooks/expected_attention.ipynb)): expected attention weight during the generation phase 
- `StreamingLLMPress` ([source](kvpress/presses/streaming_llm_press.py), [paper](https://arxiv.org/abs/2309.17453)): keep only the initial and recent tokens 
- `TOVAPress` ([source](kvpress/presses/tova_press.py), [paper](https://arxiv.org/abs/2401.06104)): attention weight of the last query averaged across heads 
- `ObservedAttentionPress` ([source](kvpress/presses/observed_attention_press.py), [paper](https://arxiv.org/abs/2306.14048)): average attention weight observed during in pre-filling phase
- `QFilterPress` ([source](kvpress/presses/qfilter_press.py), [paper](https://arxiv.org/abs/2503.02812)): project the Key representations on the main SVD component of the Query vectors to approximate the attention scores.
- `PyramidKVPress` ([source](kvpress/presses/pyramidkv_press.py), [paper](https://arxiv.org/abs/2406.02069)): maintain pyramid-like cache sizes, allocating more cache budget to lower layers and less to higher layers

Some presses rely on a different logic:
- `ThinKPress` ([source](kvpress/presses/think_press.py), [paper](https://arxiv.org/pdf/2407.21018)): compress the dimensions of the keys based on the channel attention score on the last queries 
- `SimLayerKVPress` ([source](kvpress/presses/simlayerkv_press.py), [paper](https://arxiv.org/abs/2410.13846)): identify "lazy" layers, and apply the StreamingLLM approach to them 
- `DuoAttentionPress` ([source](kvpress/presses/duo_attention_press.py), [paper](https://arxiv.org/abs/2410.10819)): split heads into retrieval heads (no compression) and streaming heads (StreamingLLM approach)
- `FinchPress` ([source](kvpress/presses/finch_press.py), [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280)): similar to SnapKV with a dynamic window size and key value re-rotation

Finally we provide wrapper presses that can be combined with other presses:
- `AdaKVPress` ([source](kvpress/presses/adakv_press.py), [paper](https://arxiv.org/abs/2407.11550)): prune bottom scores of any `ScorerPress` but across all heads, achieving head-wise compressions 
- `PerLayerCompressionPress` ([source](kvpress/presses/per_layer_compression_press.py)): compress each layer with a different compression ratio (experimental)
- `ComposedPress` ([source](kvpress/presses/composed_press.py)): compose multiple presses together by chaining their forward hooks
- `KeyRerotationPress` ([source](kvpress/presses/key_rerotation_press.py)): rerotate pruned keys to have continuous RoPE embeddings
- `ChunkKVPress` ([source](kvpress/presses/chunkkv_press.py), [paper](https://arxiv.org/abs/2502.00299)): compresses by selecting important chunks, preserving semantic coherence
- `ChunkPress` ([source](kvpress/presses/chunk_press.py), [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280)): compress the KV cache on each sequence chunk separately. This can yield to more uniform compression across long sequences
- `CriticalKVPress` and `CriticalAdaKVPress` ([source](kvpress/presses/criticalkv_press.py), [paper](https://arxiv.org/abs/2502.03805)): refine the scores using the L1 norm of Wo @ values, coupled with a two-stage selection.


For a detailed list of existing KV cache compression methods, check [Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression) or [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression?tab=readme-ov-file#kv-cache-compression)

# Tasks
- [ ] Fix RKV and RKV_LSH
- [ ] Checkpointing / Graceful run shutdown
- [ ] Confirm Attn Loss Implementation For All Presses
- [ ] Text Visualization
- [ ] Integrated Latency Code from Dixi
- [ ] Confirm SnapKV working
- [ ] Add Debug Options For All Presses
- [ ] Resolve install package ambiguities

