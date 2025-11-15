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
        self.hidden_size = None  # Will be set based on model type
        self.acc_hidden_states = None  # Will be initialized when hidden_size is known 

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
        attention_mask = torch.ones_like(attn_weights) * float("-1e9")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query_states.dtype)
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
        keys = keys / keys_norm

        # Original Algorithm: directly using the cosine similarity
        # compute the cosine similarity between keys
        keys_flat = keys.view(bsz, num_key_value_heads, -1, keys.shape[-1])
        keys_flat = keys_flat[:, :, : -self.window_size, :]  # Exclude the last window_size keys
        keys_similarity = torch.einsum("bhqd,bhkd->bhqk", keys_flat, keys_flat)
        # zero out the diagonal (self-similarity)
        mask = torch.eye(keys_similarity.shape[-1], device=keys_similarity.device).unsqueeze(0).unsqueeze(0)
        keys_similarity = keys_similarity * (1 - mask)

        redundency = keys_similarity.mean(dim=-1)  # Average over the key dimension
        redundency = F.softmax(redundency, dim=-1, dtype=torch.float32).to(scores.dtype)

        scores = scores + redundency
        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())
        return scores
    

    def _get_hidden_size(self, module, device="cuda"):
        """Get hidden size based on model type."""
        if self.hidden_size is None:
            # Detect model type from config
            model_type = getattr(module.config, 'model_type', '').lower()
            model_name = getattr(module.config, 'name_or_path', '').lower()
            
            # Check for llama3 models
            if 'llama' in model_type or 'llama' in model_name or 'nemotron' in model_name:
                self.hidden_size = 4096
            # Check for qwen-7b models
            elif 'qwen' in model_type or 'qwen' in model_name:
                if '7b' in model_name or '7b' in str(getattr(module.config, 'hidden_size', 0)):
                    self.hidden_size = 3584
                else:
                    # Default for other Qwen models
                    self.hidden_size = getattr(module.config, 'hidden_size', 4096)
            else:
                # Default: use config hidden_size or fallback to 4096
                self.hidden_size = getattr(module.config, 'hidden_size', 4096)
            
            # Initialize acc_hidden_states with correct size
            self.acc_hidden_states = torch.zeros(
                (1, self.compress_interval, self.hidden_size), dtype=torch.bfloat16, device=device
            )
        
        return self.hidden_size

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
        
        # Initialize hidden size if not set
        device = hidden_states.device
        self._get_hidden_size(module, device=device)
        
        if self.accumulated_tokens < self.compress_interval:
            if getattr(module, "layer_idx", -1) == 0:
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

        # Track token retention/eviction at first layer only
        layer_idx = getattr(module, "layer_idx", 0)
        if layer_idx == 0:
            if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self, 'input_tokens') and self.input_tokens is not None:
                try:
                    # Map position indices to actual token IDs
                    if kv_len <= len(self.input_tokens):
                        all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                        retained_positions = indices[0, 0, :, 0].cpu().tolist()  # Get retained position indices
                        retained_token_ids = [all_token_ids[pos] for pos in retained_positions]
                    else:
                        # If kv_len > input_tokens, we have generated tokens
                        all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                        retained_positions = indices[0, 0, :, 0].cpu().tolist()
                        retained_token_ids = [all_token_ids[pos] if pos < len(self.input_tokens) else pos for pos in retained_positions]
                    self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
                except Exception:
                    pass

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        # remove nan in keys and values
        keys = torch.nan_to_num(keys, nan=0.0)  
        values = torch.nan_to_num(values, nan=0.0)
        
        if layer_idx == 0:
            self.accumulated_tokens = 0  # Reset after compression
            device = hidden_states.device
            self.acc_hidden_states = torch.zeros(
                (1, self.compress_interval, self.hidden_size), dtype=torch.bfloat16, device=device
            ) # Reset accumulated hidden states
        return keys, values

