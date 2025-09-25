# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass
import os
import csv
from typing import Tuple

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class H2OPress(ScorerPress):
    """
    The h2o score is defined as the average attention weight over all prompt tokens
    Requires output_attentions=True and attn_implementation="eager" to have access to attentions
    This approach is a faithful implementation of H2O (https://arxiv.org/abs/2306.14048).
    """

    cache_budget: int = 0
    output_attentions: bool = True
    attn_csv_path: str = "attn_loss.csv"
    prune_step: int = 0

    def __post_init__(self):
        if not self.output_attentions:
            logger.warning(
                "Model will not return attentions in its output to save memory. "
                "Set output_attentions=True if attentions are needed in the output."
            )
        super().__post_init__()

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
        assert attentions is not None, 'Set output_attentions=True and attn_implementation="eager" to use this hook'
        bsz, num_key_value_heads, n_tokens, _ = keys.shape
        scores = self.acc_attn / self.n_tokens_in_sum
        scores = scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)
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
        if self.cache_budget <= 0:
            return keys, values

        # save the accumulated attention weights
        assert attentions is not None, 'Set output_attentions=True and attn_implementation="eager" to use this hook'
        bsz, n_heads, _, q_len = attentions.shape
        n_kv_groups = module.num_key_value_groups
        n_kv_heads = n_heads // n_kv_groups

        self.acc_attn = attentions.sum(2)
        # reshape attentions to bsz, n_kv_heads, n_kv_groups, q_len
        self.acc_attn = self.acc_attn.view(bsz, -1, n_kv_groups, q_len)
        # average over the n_kv_groups dimension
        self.acc_attn = self.acc_attn.mean(2) # bsz, n_kv_heads, q_len

        self.n_tokens_in_sum = torch.arange(q_len, 0, -1).to(attentions.device, attentions.dtype)
        self.n_tokens_in_sum = self.n_tokens_in_sum.unsqueeze(0).unsqueeze(0).expand(bsz, n_kv_heads, -1) # bsz, n_kv_heads, q_len

        if self.cache_budget >= q_len:
            return keys, values
  
        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, True, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices # bsz, num_key_value_heads, cache_budget

        # Prune keys and values
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim) # bsz, num_key_value_heads, cache_budget, head_dim
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()

        # Prune acc attention weights and n_tokens_in_sum
        # expand second dimension from n_kv_heads to num_heads
        self.acc_attn = self.acc_attn.gather(2, indices).contiguous()
        self.n_tokens_in_sum = self.n_tokens_in_sum.gather(2, indices).contiguous()

        return keys, values


    def compress_decoding(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.cache_budget == 0:
            return keys, values
    
        # add to the accumulated attention weights
        n_existing = self.acc_attn.shape[2]
        bsz, n_heads, _, q_len = attentions.shape

        n_kv_groups = module.num_key_value_groups
        n_kv_heads = n_heads // n_kv_groups

        new_acc_attn = attentions.sum(2) # bsz, n_heads, q_len
        # reshape attentions to bsz, n_kv_heads, n_kv_groups, q_len
        new_acc_attn = new_acc_attn.view(bsz, -1, n_kv_groups, q_len)
        # average over the n_kv_groups dimension
        new_acc_attn = new_acc_attn.mean(2) # bsz, n_kv_heads, q_len
        new_acc_attn[:, :, :n_existing] += self.acc_attn
        new_n_tokens_in_sum = torch.ones(bsz, n_kv_heads, q_len, device=attentions.device, dtype=attentions.dtype)
        new_n_tokens_in_sum[:, :, :n_existing] += self.n_tokens_in_sum
        self.acc_attn = new_acc_attn
        self.n_tokens_in_sum = new_n_tokens_in_sum

        if self.cache_budget >= q_len:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices

        # print("---" * 10)
        # print(f"[DEBUG] (PRE) keys shape: {keys.shape}, values shape: {values.shape}")
        full_len = scores.shape[-1]
        kept_len = indices.shape[2]
        # print(f"[DEBUG] diff indices: {full_len - kept_len}")

        # Compute per-head attention loss similar to RKV (pre-attn mass removed)
        try:
            bsz, n_heads, _, q_len = attentions.shape
            n_kv_groups = module.num_key_value_groups
            n_kv_heads = n_heads // n_kv_groups
            # Sum over query dimension to get per-key attention mass, then average kv groups → [B, H_kv, L]
            attn_sum = attentions.sum(2)
            attn_kv = attn_sum.view(bsz, n_kv_heads, n_kv_groups, q_len).mean(2)

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
                for head_idx in range(n_kv_heads):
                    head_attn = attn_kv[:, head_idx, :]  # [B, L]
                    kept_pos = indices[:, head_idx, :]   # [B, K]
                    pre_total = head_attn.sum(-1)        # [B]
                    kept_total = head_attn.gather(-1, kept_pos).sum(-1)  # [B]
                    loss_h = (pre_total - kept_total).sum()  # scalar over batch
                    row = [
                        self.prune_step,
                        getattr(module, "layer_idx", -1),
                        head_idx,
                        keys.shape[2],
                        full_len,
                        int(full_len - kept_len),
                        float(loss_h.item()),
                    ]
                    writer.writerow(row)
                    #print(f"[CSV] {row}")
            self.prune_step += 1
        except Exception as e:
            print(f"[WARN] H2O attn loss logging failed: {e}")

        # Prune keys and values
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim) # bsz, num_key_value_heads, cache_budget, head_dim
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()

        # print(f"[DEBUG] (PRUNED) keys shape: {keys.shape}, values shape: {values.shape}")
        # print("===" * 10)

        # Prune acc attention weights and n_tokens_in_sum
        self.acc_attn = self.acc_attn.gather(2, indices).contiguous()
        self.n_tokens_in_sum = self.n_tokens_in_sum.gather(2, indices).contiguous()

        return keys, values


    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: Tuple):
        output = super().forward_hook(module, input, kwargs, output)
        # attentions are needed as input for the hook, but unless the user wants to return them in the output,
        # we can remove them to save memory
        if not self.output_attentions:
            output = (output[0], None)

        return output
