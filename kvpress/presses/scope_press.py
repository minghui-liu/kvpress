# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.pyramidkv_press import PyramidKVPress


@dataclass
class SCOPEPress(PyramidKVPress):
    """
    SCOPE (https://arxiv.org/abs/2412.13649) compresses the KV cache separately for the prefill and decoding
    stages, since a single strategy for both stages either impairs comprehension of the full context
    (over-aggressive prefill compression) or lets the ranking of "heavy hitter" tokens go stale over long
    decoding traces ("heavy-hitter deviation").

    For the prefill stage, SCOPE reuses a pyramidal, layer-wise budget allocation (more budget in lower
    layers, less in higher ones), which is exactly what PyramidKVPress already implements, so this class
    inherits that behavior unchanged (https://github.com/Linking-ai/SCOPE/blob/main/model/kv_utils.py).

    For the decoding stage, SCOPE periodically re-ranks and re-selects the tokens generated since the last
    compression using the attention of the current query over the full cache (H2O-style), always keeping the
    most recently generated `decoding_window_size` tokens uncompressed. This periodic refresh is what
    prevents importance rankings computed during prefill (or earlier decoding steps) from becoming stale.

    We simplify the original implementation by only supporting the "slide"-style periodic re-ranking
    (SCOPE's default), instead of exposing all of "h2o" / "slide" / "adaptive" / "discontinuous" decoding
    metrics from the reference code.

    decoding_cache_budget: total number of KV pairs kept during decoding (0 disables decoding-phase
        compression, matching SCOPE's default "decoding_metric='None'").
    compress_interval: number of newly generated tokens accumulated before the cache is re-ranked and pruned
        back down to `decoding_cache_budget`.
    decoding_window_size: number of most-recently generated tokens that are never pruned.
    """

    decoding_cache_budget: int = 0
    compress_interval: int = 32
    decoding_window_size: int = 8

    def __post_init__(self):
        super().__post_init__()
        self._decoding_base_len: dict = {}

    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().compress_prefilling(module, hidden_states, keys, values, attentions, kwargs)
        # A new prefill starts a fresh decoding trace: forget any tracked decoding-compression state
        self._decoding_base_len.pop(getattr(module, "layer_idx", 0), None)
        return keys, values

    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.decoding_cache_budget <= 0:
            return keys, values

        layer_idx = getattr(module, "layer_idx", 0)
        kv_len = keys.shape[2]

        base_len = self._decoding_base_len.get(layer_idx)
        if base_len is None:
            base_len = kv_len - 1
            self._decoding_base_len[layer_idx] = base_len

        if kv_len - base_len < self.compress_interval or kv_len <= self.decoding_cache_budget:
            return keys, values

        n_keep = self.decoding_cache_budget - self.decoding_window_size
        if n_keep <= 0 or kv_len - self.decoding_window_size <= n_keep:
            self._decoding_base_len[layer_idx] = kv_len
            return keys, values

        bsz, num_key_value_heads, _, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        if attentions is not None:
            attn_weights = attentions
        else:
            attn_weights = self.compute_full_attention(module, hidden_states, keys, kwargs["position_embeddings"])

        # H2O-style importance: attention paid by the current query to every cached key so far
        scores = attn_weights.mean(dim=-2)  # (bsz, num_heads, kv_len)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, kv_len).mean(2)

        past_scores = scores[..., : -self.decoding_window_size]
        past_indices = past_scores.topk(n_keep, dim=-1).indices.sort(dim=-1).values
        # Retention tracking (head 0, layer 0): kept past positions + the always-kept
        # recent window. No-op unless tracking is enabled.
        if layer_idx == 0 and self.tokenizer is not None:
            kept = past_indices[0, 0].detach().cpu().tolist()
            window_positions = list(range(kv_len - self.decoding_window_size, kv_len))
            self.track_retained_cache_positions(kv_len, kept + window_positions)
        indices = past_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past = keys[:, :, : -self.decoding_window_size, :].gather(2, indices)
        v_past = values[:, :, : -self.decoding_window_size, :].gather(2, indices)
        k_cur = keys[:, :, -self.decoding_window_size :, :]
        v_cur = values[:, :, -self.decoding_window_size :, :]
        keys = torch.cat([k_past, k_cur], dim=2)
        values = torch.cat([v_past, v_cur], dim=2)

        self._decoding_base_len[layer_idx] = keys.shape[2]

        return keys, values

    @staticmethod
    def compute_full_attention(module, hidden_states, keys, position_embeddings):
        """
        Compute the attention of the current (single) decoding query against every cached key.
        """
        from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states)
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states)
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SCOPE not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return attn_weights
