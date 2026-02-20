# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
import time
from kvpress.presses.scorer_press import ScorerPress

GLOBAL_ATTN_WEIGHTS = None

@dataclass
class RKVPress(ScorerPress):
    """
    RKV (https://www.arxiv.org/pdf/2505.24133)
    """

    cache_budget: int = 0
    compress_interval: int = 64 # aka. the buffer size 
    # compression_ratio: float = 0.0
    window_size: int = 64 # number of observation tokens always kept in the cache
    kernel_size: int = 5
    prune_step: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.accumulated_tokens = 0  # Initialize accumulated tokens for compression interval
        self.acc_hidden_states = torch.zeros(
            (1, self.compress_interval, 3584), dtype=torch.bfloat16, device="cuda"
        )  # Initialize accumulated hidden states 
        self._acc_hsize = 3584

    def reset_timing(self):
        """Reset timing counters and internal accumulation state"""
        super().reset_timing()
        self.accumulated_tokens = 0
        # Re-initialize buffer to avoid stale hidden states
        self.acc_hidden_states = torch.zeros(
            (1, self.compress_interval, self._acc_hsize),
            dtype=self.acc_hidden_states.dtype,
            device=self.acc_hidden_states.device,
        )

    def _try_resize_buffer(self, hidden_states: torch.Tensor):
        """If buffer shape mismatches, try fallback hidden sizes in order: 3584, 5120, 4096.
        If one matches hidden_states last dim, allocate and return True. Else return False.
        """
        hs = hidden_states.shape[-1]
        for cand in (3584, 5120, 4096):
            if hs == cand:
                self.acc_hidden_states = torch.zeros(
                    (hidden_states.shape[0], self.compress_interval, cand),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                self._acc_hsize = cand
                self.accumulated_tokens = 0
                return True
        return False

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

        global GLOBAL_ATTN_WEIGHTS
        GLOBAL_ATTN_WEIGHTS = attn_weights.detach()

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
        keys = keys / keys_norm

        # compute the cosine similarity between keys
        keys_flat = keys.view(bsz, num_key_value_heads, -1, keys.shape[-1])
        keys_flat = keys_flat[:, :, : -self.window_size, :]  # Exclude the last window_size keys
        keys_similarity = torch.einsum("bhqd,bhkd->bhqk", keys_flat, keys_flat)
        # zero out the diagonal (self-similarity)
        mask = torch.eye(keys_similarity.shape[-1], device=keys_similarity.device).unsqueeze(0).unsqueeze(0)
        keys_similarity = keys_similarity * (1 - mask)

        redundency = keys_similarity.mean(dim=-1)  # Average over the key dimension
        redundency = F.softmax(redundency, dim=-1, dtype=torch.float32).to(scores.dtype)
 
        lam = 0
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
        layer_idx = getattr(module, "layer_idx", -1)
        
        if self.cache_budget >= kv_len:
            # All tokens retained - no compression needed, skip tracking to avoid excessive entries
            return keys, values
        
        if self.accumulated_tokens < self.compress_interval:
            if layer_idx == 0:
                self.accumulated_tokens += 1
            
            if self.debug:
                print(f"[DEBUG] (STEP) accumulated_tokens: {self.accumulated_tokens} / {self.compress_interval}")
            
            try:
                self.acc_hidden_states[:, self.accumulated_tokens - 1, :] = hidden_states
            except RuntimeError:
                if not self._try_resize_buffer(hidden_states):
                    # As a last resort, allocate with the actual hidden size
                    hs = hidden_states.shape[-1]
                    self.acc_hidden_states = torch.zeros(
                        (hidden_states.shape[0], self.compress_interval, hs),
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    self._acc_hsize = hs
                    self.accumulated_tokens = 0
                # write after successful resize (accumulated_tokens was reset to 0 then incremented above)
                self.acc_hidden_states[:, self.accumulated_tokens - 1, :] = hidden_states
            return keys, values

        # Compute scores using the buffered window
        scores = self.score(
            module,
            self.acc_hidden_states[:, -self.window_size:, :],
            keys,
            values,
            attentions,
            False,
            kwargs,
        )
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices

        # Attention-based loss logging (use real attentions only if provided)
        if not self.latency:
            self.compute_attention_loss(module, attentions, indices, window_size=self.window_size)

        # Track token retention/eviction at first layer only
        if layer_idx == 0:
            if self.debug:
                print(f"[TRACK DEBUG] layer_idx=0, tokenizer={self.tokenizer is not None}, input_tokens={self.input_tokens is not None}")
            if self.tokenizer is not None and self.input_tokens is not None:
                # Map position indices to actual token IDs
                retained_positions = indices[0, 0, :].cpu().tolist()  # Get retained position indices
                
                # Extract importance scores for all positions (SUM across heads)
                # scores shape: [batch, num_heads, kv_len] -> sum across heads
                # Using sum so evicted tokens (which get 0 in some heads) show lower total
                importance_scores = scores[0].sum(dim=0).cpu().tolist()  # [kv_len]
                
                if kv_len <= len(self.input_tokens):
                    all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                    retained_token_ids = [all_token_ids[pos] for pos in retained_positions if pos < len(all_token_ids)]
                else:
                    # If kv_len > input_tokens, we have generated tokens
                    all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                    retained_token_ids = [all_token_ids[pos] if pos < len(all_token_ids) else pos for pos in retained_positions]
                
                self.track_generation_step(
                    all_token_ids, 
                    retained_token_ids, 
                    self.tokenizer,
                    scores=importance_scores,
                    retained_positions=retained_positions
                )
                if self.debug:
                    print(f"[TRACK DEBUG] Tracked step, generation_steps count: {len(self.generation_steps)}")

        # expand for gather
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        if layer_idx == 0:
            # Reset after compression; keep current detected hidden size or try fallbacks
            self.accumulated_tokens = 0
            hs = getattr(self, "_acc_hsize", 3584)
            if hs not in (3584, 5120, 4096):
                # if an unusual size was detected earlier, keep using it
                pass
            self.acc_hidden_states = torch.zeros(
                (1, self.compress_interval, hs), dtype=hidden_states.dtype, device=hidden_states.device
            )
        
        if self.debug:
            print(f"[DEBUG] (PRUNED) keys length: {keys.shape[2]}")

        return keys, values

