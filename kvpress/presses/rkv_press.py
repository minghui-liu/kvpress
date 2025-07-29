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
class RKVPress(ScorerPress):
    """
    RKV (https://www.arxiv.org/pdf/2505.24133)
    """

    cache_budget: int = 0
    compress_interval: int = 64 # aka. the buffer size 
    # compression_ratio: float = 0.0
    window_size: int = 8 # number of observation tokens always kept in the cache
    kernel_size: int = 5

    def __post_init__(self):
        super().__post_init__()
        self.accumulated_tokens = 0  # Initialize accumulated tokens for compression interval
        self.acc_hidden_states = torch.zeros(
            (1, self.compress_interval, 4096), dtype=torch.bfloat16, device="cuda"
        )  # Initialize accumulated hidden states 

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
        
        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

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
        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.max(dim=-2).values
        # Stablization and Importance Estimation
        scores = F.max_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        # Redundancy Estimation via Semantic Similarity
        
        # normalize keys by dividing the l2 norm of keys + eps (1e-8) 
        eps = 1e-8
        keys_norm = keys.norm(dim=-1, keepdim=True) + eps
        keys_norm = torch.clamp(keys.norm(dim=-1, keepdim=True), min=1e-6)
        keys = keys / keys_norm

        ### Original Algorithm: directly using the cosine similarity
        # # compute the cosine similarity between keys
        # keys_flat = keys.view(bsz, num_key_value_heads, -1, keys.shape[-1])
        # keys_flat = keys_flat[:, :, : -self.window_size, :]  # Exclude the last window_size keys
        # keys_similarity = torch.einsum("bhqd,bhkd->bhqk", keys_flat, keys_flat)
        # # zero out the diagonal (self-similarity)
        # mask = torch.eye(keys_similarity.shape[-1], device=keys_similarity.device).unsqueeze(0).unsqueeze(0)
        # keys_similarity = keys_similarity * (1 - mask)

        # redundency = keys_similarity.mean(dim=-1)  # Average over the key dimension
        # redundency = F.softmax(redundency, dim=-1, dtype=torch.float32).to(scores.dtype)
 

        ### Modified Algorithm: implement LSH over that
        keys_flat = keys.view(bsz, num_key_value_heads, -1, keys.shape[-1])
        keys_flat = keys_flat[:, :, : -self.window_size, :]  # Exclude the last window_size keys

        # Construct LSH buckets
        n_hash_buckets=16
        proj_matrix = torch.randn(keys_flat.shape[-1],n_hash_buckets, device=keys.device).to(keys_flat.dtype)  # Random projection matrix
        # Dixi: I use random projection here has hash function for easiest implementation
        hash_bits = torch.einsum("bhqd,dk->bhqk", keys_flat, proj_matrix)
        hash_codes = (hash_bits > 0).int()
        powers_of_two = 2 ** torch.arange(n_hash_buckets, device=keys.device, dtype=torch.float32)
        hash_codes_int = torch.sum(hash_codes * powers_of_two, dim=-1)  # [B, H, Q]

        redundency = torch.zeros_like(hash_codes_int, dtype=torch.float32)  # [B, H, Q]
        for b in range(bsz):
            for h in range(num_key_value_heads):
                codes = hash_codes_int[b, h]  # [Q]
                unique, counts = torch.unique(codes, return_counts=True)
                count_dict = dict(zip(unique.tolist(), counts.tolist()))
                redundency[b, h] = torch.tensor([count_dict[c.item()] for c in codes], device=keys.device)
        
        redundency = torch.clamp(redundency, min=1.0)  # ensure no zero or negative
        redundency = F.softmax(redundency, dim=-1, dtype=torch.float32).to(scores.dtype)


        lam = 0.1
        scores = lam * scores + (1 - lam) * redundency
        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())
        return scores
    

    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_budget == 0:
            return keys, values

        kv_len = keys.shape[2]
        if self.cache_budget >= kv_len:
            return keys, values
        
        if self.accumulated_tokens < self.compress_interval:
            self.accumulated_tokens += 1
            # # print(f"[DEBUG] hidden_states shape: {hidden_states.shape}, acc_hidden_states shape: {self.acc_hidden_states.shape}, accumulated_tokens: {self.accumulated_tokens}")
            self.acc_hidden_states[:, self.accumulated_tokens - 1, :] = hidden_states
            return keys, values

        # Compute scores
        # scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        scores = self.score(module, self.acc_hidden_states[:, -self.window_size:, :], keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        self.accumulated_tokens = 0  # Reset after compression
        self.acc_hidden_states = torch.zeros(
            (1, self.compress_interval, 4096), dtype=torch.bfloat16, device="cuda"
        ) # Reset accumulated hidden states
        return keys, values

