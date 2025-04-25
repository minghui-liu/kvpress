# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class ScorerPress(BasePress):
    """
    Default press method for using a score method.
    Any ScorerPress subclass must implement the `score` method that computes a tensor of scores for each key-value pair
    The KV pairs with the lowest scores will be pruned in the `compress` method.
    The cache is uniformly pruned across all heads and layers using the compression_ratio parameter.
    """

    compression_ratio: float = 0.0
    cache_budget: int = 0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute a tensor of scores with shape (bsz, num_key_value_heads, q_len)
        The KV pairs with lowest scores will be pruned in the `compress` method.
        """
        raise NotImplementedError

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

        q_len = hidden_states.shape[1]
        if self.cache_budget >= q_len:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

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

        if self.cache_budget == 0:
            return keys, values

        q_len = hidden_states.shape[1]
        if self.cache_budget >= q_len:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        
        # Get index of KV pairs with the lowest score
        index_to_remove = scores.argmin(scores, dim=-1)
        index_to_remove = index_to_remove.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = torch.cat([
                keys[:, :index_to_remove, :],
                keys[:, index_to_remove + 1 :, :],
            ], dim=1)
        values = torch.cat([
                values[:, :index_to_remove, :],
                values[:, index_to_remove + 1 :, :],
            ], dim=1)
        return keys, values
