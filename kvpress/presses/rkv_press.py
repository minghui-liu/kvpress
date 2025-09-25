# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import os
import csv
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

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
    window_size: int = 8 # number of observation tokens always kept in the cache
    kernel_size: int = 5
    attn_csv_path: str = "attn_loss.csv"
    prune_step: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.accumulated_tokens = 0  # Initialize accumulated tokens for compression interval
        self.acc_hidden_states = torch.zeros(
            (1, self.compress_interval, 3584), dtype=torch.bfloat16, device="cuda"
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

        global GLOBAL_ATTN_WEIGHTS
        GLOBAL_ATTN_WEIGHTS = attn_weights.detach()

        # print(f"[DEBUG] attn_weights shape: {attn_weights.shape}")

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

        lam = 0.1
        scores = lam * scores + (1 - lam) * redundency

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores


    def get_avg_attention_for_index(indices, attn_weights, b=0, k=0):
        # Sum attention over all query heads and window queries for the selected key index k per KV head
        # Collapse expanded indices if needed: [bsz, kv_heads, budget, head_dim] -> [kv_heads, budget]
        idx = indices[b, :, :, 0]

        # Use high precision for accumulation
        attn_weights = attn_weights.to(torch.float32)
        max_valid_key = attn_weights.shape[-1]  # q_len - window_size
        total = torch.zeros((), device=attn_weights.device, dtype=torch.float32)
        for kvh in range(idx.shape[0]):
            key_idx = int(idx[kvh, k].item())
            if key_idx >= max_valid_key:
                continue
            total = total + attn_weights[b, :, :, key_idx].sum()

        return total

    def get_avg_attention_for_all_indices(indices, attn_weights, b=0):
        # Use high precision for accumulation
        attn_weights = attn_weights.to(torch.float32)
        total = torch.zeros((), device=attn_weights.device, dtype=torch.float32)
        num_positions = indices.shape[2]
        for k in range(num_positions):
            total = total + RKVPress.get_avg_attention_for_index(indices, attn_weights, b=b, k=k)
        return total


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
            """
            print(f"---"*10)
            print(f"[DEBUG] (STEP) keys shape: {keys.shape}, values shape: {values.shape}")
            step_attn_len = max(keys.shape[2] - self.window_size, 0)
            print(f"[DEBUG] (STEP) expected attn last dim: {step_attn_len}")
            print(f"[DEBUG] (STEP) accumulated_tokens: {self.accumulated_tokens + 1} / {self.compress_interval}")
            """

            if getattr(module, "layer_idx", -1) == 0:
                self.accumulated_tokens += 1
            
            #print(f"[DEBUG] (STEP) accumulated_tokens: {self.accumulated_tokens} / {self.compress_interval}")
            # # print(f"[DEBUG] hidden_states shape: {hidden_states.shape}, acc_hidden_states shape: {self.acc_hidden_states.shape}, accumulated_tokens: {self.accumulated_tokens}")
            self.acc_hidden_states[:, self.accumulated_tokens - 1, :] = hidden_states
            return keys, values

        # Compute scores
        # scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        scores = self.score(module, self.acc_hidden_states[:, -self.window_size:, :], keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores

        # print(f"---"*10)
        # print(f"[DEBUG] (PRE) keys shape: {keys.shape}, values shape: {values.shape}")
        full_indices = scores.topk(scores.shape[-1], dim=-1).indices
        full_indices = full_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        diff_count = full_indices.shape[2] - indices.shape[2]
        attn_loss_value = (
            RKVPress.get_avg_attention_for_all_indices(full_indices, GLOBAL_ATTN_WEIGHTS, b=0)
            - RKVPress.get_avg_attention_for_all_indices(indices, GLOBAL_ATTN_WEIGHTS, b=0)
        )
        # print(f"[DEBUG] diff indices: {diff_count}")
        # print(f"[DEBUG] attn loss: {attn_loss_value}")
        # print(f"[DEBUG] attn size: {GLOBAL_ATTN_WEIGHTS.shape}")
        # Append per-head rows to CSV (store layer and head indices)
        csv_path = self.attn_csv_path
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "prune_step",
                    "layer_idx",
                    "head_idx",
                    "kv_len_pre",
                    "attn_len",
                    "diff_indices",
                    "attn_loss",
                ])
            bsz, num_kv_heads, _, _ = indices.shape
            for head_idx in range(num_kv_heads):
                # Slice indices for this head only
                full_h = full_indices[:, head_idx : head_idx + 1, :, :]
                kept_h = indices[:, head_idx : head_idx + 1, :, :]
                # Compute per-head loss with the same method
                loss_h = (
                    RKVPress.get_avg_attention_for_all_indices(full_h, GLOBAL_ATTN_WEIGHTS, b=0)
                    - RKVPress.get_avg_attention_for_all_indices(kept_h, GLOBAL_ATTN_WEIGHTS, b=0)
                )
                row = [
                    self.prune_step,
                    getattr(module, "layer_idx", -1),
                    head_idx,
                    keys.shape[2],
                    GLOBAL_ATTN_WEIGHTS.shape[-1] if GLOBAL_ATTN_WEIGHTS is not None else -1,
                    full_h.shape[2] - kept_h.shape[2],
                    float(loss_h.item() if hasattr(loss_h, "item") else float(loss_h)),
                ]
                writer.writerow(row)
                #print(f"[CSV] {row}")
        self.prune_step += 1
        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        # print(f"[DEBUG] (PRUNED) keys shape: {keys.shape}, values shape: {values.shape}")
        # print(f"==="*10)

        if getattr(module, "layer_idx", -1) == 0:
            self.accumulated_tokens = 0  # Reset after compression
        self.acc_hidden_states = torch.zeros(
            (1, self.compress_interval, 3584), dtype=torch.bfloat16, device="cuda"
        ) # Reset accumulated hidden states
        return keys, values