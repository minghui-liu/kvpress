# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class RPCPress(ScorerPress):
    """
    Reasoning Path Compression (RPC, https://arxiv.org/abs/2505.13866,
    https://github.com/jiwonsong-dev/ReasoningPathCompression) is a training-free, decoding-time KV cache
    compression method targeting long chain-of-thought reasoning traces.

    RPC never compresses the prefill/prompt: the full prompt is always kept in the cache. During decoding, it
    periodically prunes the tokens generated so far ("the reasoning path"), keeping only the ones with the
    highest attention-based importance while always keeping the most recently generated `window_size` tokens
    uncompressed. Importance is estimated from a small "selector window" of the most recent decoding queries
    (https://github.com/jiwonsong-dev/ReasoningPathCompression/blob/main/rpc/rpc_utils.py), which is cheap to
    compute and, per the paper, tracks which earlier reasoning steps remain relevant better than a single
    query would.

    Every `compress_interval` newly generated tokens, RPC grows the retained budget for the reasoning path by
    `compress_interval` tokens (up to `cache_budget`), re-selecting the tokens to keep from scratch. This
    mirrors the original method's monotonically growing budget, but caps it at `cache_budget` so the cache
    size stays bounded for arbitrarily long generations.

    We simplify the original implementation by always using the "recent queries" selector with "all-heads"
    attention aggregation (RPC's default configuration), instead of exposing the "prompt" / "new" selector
    and "group" / "none" aggregation options from the reference code.
    """

    cache_budget: int = 0
    window_size: int = 32
    compress_interval: int = 128
    kernel_size: int = 7

    def __post_init__(self):
        super().__post_init__()
        self._prompt_len: dict = {}
        self._n_selected: dict = {}
        self._recent_hidden: dict = {}

    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # RPC never compresses the prompt itself, it only prunes tokens generated during decoding
        layer_idx = getattr(module, "layer_idx", 0)
        self._prompt_len[layer_idx] = keys.shape[2]
        self._n_selected[layer_idx] = 0
        self._recent_hidden[layer_idx] = []
        # Prompt is fully retained: record 100% prompt retention (layer 0, no-op unless tracking on).
        if layer_idx == 0 and self.tokenizer is not None:
            self.track_retained_cache_positions(keys.shape[2], list(range(keys.shape[2])))
        return keys, values

    @staticmethod
    def compute_selector_attention(module, hidden_buffer, keys, prompt_len, window_size):
        """
        Compute the attention of a buffer of recent (already-projected-position) decoding queries against the
        "compressible" part of the cache (i.e. excluding the prompt and the most recent window_size tokens).
        Each buffer entry stores the hidden state produced at that decoding step together with the RoPE
        cos/sin for that step's position, so every query is rotated with its own position.
        """
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        query_list = []
        for hidden_state, (cos, sin) in hidden_buffer:
            bsz = hidden_state.shape[0]
            if hasattr(module, "q_proj"):
                query_states = module.q_proj(hidden_state)
            elif hasattr(module, "qkv_proj"):
                qkv = module.qkv_proj(hidden_state)
                query_states = qkv[..., : num_heads * head_dim]
            else:
                raise NotImplementedError(f"RPCPress not yet implemented for {module.__class__}.")

            query_states = query_states.view(bsz, 1, num_heads, head_dim).transpose(1, 2)
            query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))
            query_list.append(query_states)
        query_states = torch.cat(query_list, dim=2)  # (bsz, num_heads, len(buffer), head_dim)

        key_states = repeat_kv(keys[:, :, prompt_len:-window_size, :], num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return attn_weights

    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.cache_budget <= 0:
            return keys, values

        layer_idx = getattr(module, "layer_idx", 0)
        buffer = self._recent_hidden.setdefault(layer_idx, [])
        buffer.append((hidden_states, kwargs["position_embeddings"]))
        if len(buffer) > self.window_size:
            buffer.pop(0)

        prompt_len = self._prompt_len.get(layer_idx, 0)
        n_selected = self._n_selected.get(layer_idx, 0)
        kv_len = keys.shape[2]

        if n_selected >= self.cache_budget:
            return keys, values

        target_len = prompt_len + n_selected + self.window_size
        if kv_len < target_len + self.compress_interval:
            return keys, values

        n_keep = min(n_selected + self.compress_interval, self.cache_budget)

        bsz, num_key_value_heads, _, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        attn_weights = self.compute_selector_attention(module, buffer, keys, prompt_len, self.window_size)
        scores = attn_weights.mean(dim=-2)  # (bsz, num_heads, kv_len - prompt_len - window_size)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, -1).mean(2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        mid_indices = scores.topk(n_keep, dim=-1).indices.sort(dim=-1).values
        # Retention tracking (head 0, layer 0): full prompt + kept reasoning tokens +
        # always-kept recent window. No-op unless tracking is enabled.
        if layer_idx == 0 and self.tokenizer is not None:
            mid = [prompt_len + i for i in mid_indices[0, 0].detach().cpu().tolist()]
            retained = list(range(prompt_len)) + mid + list(range(kv_len - self.window_size, kv_len))
            self.track_retained_cache_positions(kv_len, retained)
        indices = mid_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_prompt = keys[:, :, :prompt_len, :]
        v_prompt = values[:, :, :prompt_len, :]
        k_mid = keys[:, :, prompt_len:-self.window_size, :].gather(2, indices)
        v_mid = values[:, :, prompt_len:-self.window_size, :].gather(2, indices)
        k_recent = keys[:, :, -self.window_size :, :]
        v_recent = values[:, :, -self.window_size :, :]

        keys = torch.cat([k_prompt, k_mid, k_recent], dim=2)
        values = torch.cat([v_prompt, v_mid, v_recent], dim=2)

        self._n_selected[layer_idx] = n_keep
        self._recent_hidden[layer_idx] = []

        return keys, values
