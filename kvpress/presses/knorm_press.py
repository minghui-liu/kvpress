# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KnormPress(ScorerPress):
    """Prune KV pairs with highest L2 norm of keys (https://arxiv.org/pdf/2406.11430)"""

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

        # Compute scores and top-k indices
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        indices = scores.topk(self.cache_budget, dim=-1).indices
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Attention-based loss logging (match H2O behavior)
        if not self.latency:
            self.compute_attention_loss(module, attentions, indices, window_size=0)

        # Prune keys and values
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()
        return keys, values
