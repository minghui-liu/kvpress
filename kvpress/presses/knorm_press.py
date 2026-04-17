# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
import os
import csv

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KnormPress(ScorerPress):
    """Prune KV pairs with highest L2 norm of keys (https://arxiv.org/pdf/2406.11430)"""

    attn_csv_path: str = "attn_loss.csv"
    prune_step: int = 0

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
        return -keys.norm(dim=-1)

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

        # Compute scores with L2 norm (more negative = more important)
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        indices = scores.topk(self.cache_budget, dim=-1).indices  # [B, Hkv, K]
        full_len = int(scores.shape[-1])
        kept_len = int(indices.shape[2])


        # Debug counts
        # try:
        #     print("---" * 10)
        #     print(f"[DEBUG] (PRE) keys shape: {keys.shape}, values shape: {values.shape}")
        #     full_len = scores.shape[-1]
        #     kept_len = indices.shape[2]
        #     print(f"[DEBUG] diff indices: {full_len - kept_len}")
        # except Exception:
        #     pass

        # If attentions are provided, compute attention mass removed (pre-attn only)
        if 1:
            bsz, n_heads, _, q_len = attentions.shape
            n_kv_groups = module.num_key_value_groups
            n_kv_heads = n_heads // n_kv_groups
            attn_sum = attentions.sum(2)  # [B, H, L]
            attn_kv = attn_sum.view(bsz, n_kv_heads, n_kv_groups, q_len).mean(2)  # [B, Hkv, L]

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
                for head_idx in range(attn_kv.shape[1]):
                    head_attn = attn_kv[:, head_idx, :]  # [B, L]
                    kept_pos = indices[:, head_idx, :]    # [B, K]
                    pre_total = head_attn.sum(-1)         # [B]
                    kept_total = head_attn.gather(-1, kept_pos).sum(-1)  # [B]
                    loss_h = (pre_total - kept_total).sum()  # scalar over batch
                    row = [
                        self.prune_step,
                        getattr(module, "layer_idx", -1),
                        head_idx,
                        keys.shape[2],
                        q_len,
                        int(full_len - kept_len),
                        float(loss_h.item()),
                    ]
                    writer.writerow(row)
                        # print(f"[CSV] {row}")
            self.prune_step += 1

        # Gather pruned keys/values
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()
        return keys, values
