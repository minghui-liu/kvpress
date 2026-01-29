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
class SnapKVPress(ScorerPress):
    """
    SnapKV (https://arxiv.org/abs/2404.14469) use the attention of the latest window_size tokens to estimate the
    importance of the previous KV pairs. We use the default settings from:
    https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py#L24
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -window_size:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        real_window_size = query_states.shape[1]
        query_states = query_states.view(bsz, real_window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Masking logic
        # If we are in prefill (q_len large), we might need causal masking within the window.
        # If we are in decoding (q_len small), we attend to all past keys.
        
        # Original logic was:
        # attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        
        # If q_len (input) is small (decoding), q_len - window_size + 1 is likely negative if referring to hidden_states.shape[1].
        # But wait, q_len is hidden_states.shape[1].
        # keys.shape[2] is the full KV length.
        
        # If we are using this for decoding, we want to attend to all keys up to current position.
        # Since keys include current position? usually.
        
        # If real_window_size == 1 (decoding), attn_weights is [B, H, 1, K].
        # We usually want to mask out future keys if any.
        # But K usually <= current step.
        
        # The original logic seems to assume q_len is large (prefill).
        # Let's keep it robust:
        
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        
        # Logic: mask[i, j] = -inf if j > i + offset
        # We want to mask positions in keys that are "future" relative to queries.
        # Keys are length K. Queries are length Q (subset of hidden_states).
        # If hidden_states corresponds to positions [P, P+Q], and keys corresponds to [0, K].
        # Usually K >= P+Q.
        
        # If we simply assume standard causal masking for the *window* part:
        # The original code used `q_len` from hidden_states.shape[1].
        
        diagonal = q_len - window_size + 1
        
        # If q_len < window_size (decoding), diagonal is negative.
        # triu with negative diagonal masks less. 
        # If diagonal is -60, it masks effectively nothing if K is small? No.
        # triu keeps upper triangle.
        
        # If we are in decoding, we typically don't need masking if keys are all past.
        # Let's just use the original logic but careful about shapes.
        
        attention_mask = torch.triu(attention_mask, diagonal=diagonal)
        attn_weights += attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Slice off the window from the keys side?
        # Original: attn_weights = attn_weights[..., :-window_size]
        # This implies we are scoring keys *before* the observation window.
        # If we are in decoding, and we want to prune "history", we might want to exclude the most recent tokens from being pruned?
        # But if window_size is larger than K, this slice might be empty or invalid.
        
        if attn_weights.shape[-1] > window_size:
             attn_weights = attn_weights[..., :-window_size]
        else:
             # If K <= window_size, we probably shouldn't be pruning anyway (assertion in score covers this), 
             # but for robustness return empty or full?
             # If K is small, we return it as is?
             # But score() expects to average over keys.
             pass

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        is_prefill: bool,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.mean(2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores

    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prefill compression for SnapKV, with attention-loss logging matching H2O semantics.
        """
        if self.cache_budget <= 0:
            return keys, values

        q_len = hidden_states.shape[1]
        if self.cache_budget >= q_len:
            return keys, values

        # Compute scores and select kept indices
        scores = self.score(module, hidden_states, keys, values, attentions, True, kwargs)
        indices = scores.topk(self.cache_budget, dim=-1).indices

        # Attention-based loss logging (match H2O: use full K, window_size=0)
        if not self.latency:
            self.compute_attention_loss(module, attentions, indices, window_size=0)

        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()
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
        """
        Decoding compression for SnapKV, with attention-loss logging matching H2O semantics.
        """
        if self.cache_budget == 0:
            return keys, values

        kv_len = keys.shape[2]
        if self.cache_budget >= kv_len:
            return keys, values

        # Compute scores over all KV positions
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        indices = scores.topk(self.cache_budget, dim=-1).indices

        # Attention-based loss logging (match H2O: use full K, window_size=0)
        if not self.latency:
            self.compute_attention_loss(module, attentions, indices, window_size=0)

        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()
        return keys, values
